"""CUDA batch kernel (CuPy RawModule). Same names and semantics as ``_core_py``
(the spec), batch functions only: single elements stay on the CPU, where they
take ~1 us against ~10 us for a kernel launch alone.

Precision: float32 by default (GeForce/Jetson FP64 runs at 1/32..1/64 of FP32),
float64 on request, mainly for parity checks. Series thresholds depend on the
precision, see SMALL_ANGLE / NEAR_PI.

Data placement: NumPy in -> float64 NumPy out (drop-in for the CPU kernels,
computed in the chosen dtype). Any CuPy input -> CuPy out in the compute
dtype: data stays on the device.

Like the CPU kernels, these functions are UNCHECKED.
"""
from __future__ import annotations

import numpy as np

from . import _core_py

try:
    import cupy as cp
except ImportError as exc:
    raise ImportError("CUDA backend needs CuPy: pip install 'cupy-cuda12x[ctk]'") from exc

try:
    _N_DEVICES = cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError as exc:
    raise ImportError(f"CUDA backend: no usable CUDA driver ({exc})") from exc
if _N_DEVICES < 1:
    raise ImportError("CUDA backend: no CUDA device")

# float32 closed forms like (t - sin t)/t^3 lose most digits at t ~ 1e-3;
# at 5e-2 the error is ~1 ulp and the truncated series is still exact to ~1e-11.
SMALL_ANGLE = {"float32": 5e-2, "float64": 1e-3}
NEAR_PI = {"float32": 5e-2, "float64": 1e-3}

_F32, _F64 = np.dtype(np.float32), np.dtype(np.float64)
_CTYPE = {_F32: "float", _F64: "double"}

_SOURCE = r"""
#define RD __device__ __forceinline__
#define GRID_LOOP(i, n) \
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
         i += (long long)blockDim.x * gridDim.x)

RD float  m_sin(float x)  { return sinf(x); }   RD double m_sin(double x)  { return sin(x); }
RD float  m_cos(float x)  { return cosf(x); }   RD double m_cos(double x)  { return cos(x); }
RD float  m_tan(float x)  { return tanf(x); }   RD double m_tan(double x)  { return tan(x); }
RD float  m_sqrt(float x) { return sqrtf(x); }  RD double m_sqrt(double x) { return sqrt(x); }
RD float  m_fabs(float x) { return fabsf(x); }  RD double m_fabs(double x) { return fabs(x); }
RD float  m_atan2(float y, float x)   { return atan2f(y, x); }
RD double m_atan2(double y, double x) { return atan2(y, x); }

template <typename T> struct Thr;
template <> struct Thr<float> {
    static RD float small()   { return @SMALL32@f; }
    static RD float near_pi() { return @NEARPI32@f; }
};
template <> struct Thr<double> {
    static RD double small()   { return @SMALL64@; }
    static RD double near_pi() { return @NEARPI64@; }
};

template <typename T> RD void coeffs(T t, T& a, T& b, T& c) {
    if (t < Thr<T>::small()) {
        T t2 = t * t, t4 = t2 * t2;
        a = T(1) - t2 / T(6) + t4 / T(120);
        b = T(0.5) - t2 / T(24) + t4 / T(720);
        c = T(1) / T(6) - t2 / T(120) + t4 / T(5040);
    } else {
        T s = m_sin(t), h = m_sin(T(0.5) * t);
        a = s / t;
        b = T(2) * h * h / (t * t);
        c = (t - s) / (t * t * t);
    }
}

template <typename T> RD T d_coeff(T t) {
    if (t < Thr<T>::small()) {
        T t2 = t * t;
        return T(1) / T(12) + t2 / T(720) + t2 * t2 / T(30240);
    }
    T half = T(0.5) * t;
    return (T(1) - half / m_tan(half)) / (t * t);
}

template <typename T> RD void hat(const T* w, T* k) {
    k[0] = T(0);  k[1] = -w[2]; k[2] = w[1];
    k[3] = w[2];  k[4] = T(0);  k[5] = -w[0];
    k[6] = -w[1]; k[7] = w[0];  k[8] = T(0);
}

// out = c0*I + c1*K + c2*K^2 using K^2 = w w^T - t2 I
template <typename T> RD void poly(const T* w, T t2, T c0, T c1, T c2, T* out, int ld) {
    T k[9];
    hat(w, k);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) out[i * ld + j] = c1 * k[i * 3 + j] + c2 * w[i] * w[j];
        out[i * ld + i] += c0 - c2 * t2;
    }
}

template <typename T> RD void so3_log(const T* r, int ld, T* w) {
    T v[3];
    v[0] = T(0.5) * (r[2 * ld + 1] - r[1 * ld + 2]);
    v[1] = T(0.5) * (r[0 * ld + 2] - r[2 * ld + 0]);
    v[2] = T(0.5) * (r[1 * ld + 0] - r[0 * ld + 1]);
    T sin_t = m_sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    T cos_t = T(0.5) * (r[0] + r[ld + 1] + r[2 * ld + 2] - T(1));
    if (cos_t > T(1)) cos_t = T(1);
    else if (cos_t < T(-1)) cos_t = T(-1);
    T theta = m_atan2(sin_t, cos_t);

    if (theta < Thr<T>::small()) {
        T t2 = theta * theta;
        T f = T(1) + t2 / T(6) + T(7) * t2 * t2 / T(360);
        for (int i = 0; i < 3; ++i) w[i] = f * v[i];
        return;
    }
    if (T(3.14159265358979323846) - theta < Thr<T>::near_pi()) {
        // symmetric part: (R + R^T)/2 = cos(t) I + (1 - cos(t)) a a^T
        T m[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) m[i * 3 + j] = T(0.5) * (r[i * ld + j] + r[j * ld + i]);
            m[i * 3 + i] -= cos_t;
        }
        for (int i = 0; i < 9; ++i) m[i] /= (T(1) - cos_t);
        int k = 0;
        if (m[4] > m[k * 3 + k]) k = 1;
        if (m[8] > m[k * 3 + k]) k = 2;
        T s = m[k * 3 + k];
        if (s < T(0)) s = T(0);
        s = m_sqrt(s);
        for (int i = 0; i < 3; ++i) w[i] = m[k * 3 + i] / s;
        T n = m_sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        if (w[0] * v[0] + w[1] * v[1] + w[2] * v[2] < T(0)) n = -n;
        for (int i = 0; i < 3; ++i) w[i] = theta * w[i] / n;
        return;
    }
    T f = theta / sin_t;
    for (int i = 0; i < 3; ++i) w[i] = f * v[i];
}

template <typename T> RD void se3_exp(const T* xi, T* t) {
    T w[3] = {xi[0], xi[1], xi[2]};
    T v[3] = {xi[3], xi[4], xi[5]};
    T t2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    T a, b, c, vm[9];
    coeffs(m_sqrt(t2), a, b, c);
    poly(w, t2, T(1), a, b, t, 4);
    poly(w, t2, T(1), b, c, vm, 3);
    for (int i = 0; i < 3; ++i)
        t[i * 4 + 3] = vm[i * 3] * v[0] + vm[i * 3 + 1] * v[1] + vm[i * 3 + 2] * v[2];
    t[12] = T(0); t[13] = T(0); t[14] = T(0); t[15] = T(1);
}

template <typename T> RD void se3_log(const T* t, T* xi) {
    T w[3], vinv[9];
    so3_log(t, 4, w);
    T t2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    poly(w, t2, T(1), T(-0.5), d_coeff(m_sqrt(t2)), vinv, 3);
    for (int i = 0; i < 3; ++i) {
        xi[i] = w[i];
        xi[3 + i] = vinv[i * 3] * t[3] + vinv[i * 3 + 1] * t[7] + vinv[i * 3 + 2] * t[11];
    }
}

template <typename T> RD void se3_inv(const T* t, T* o) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) o[i * 4 + j] = t[j * 4 + i];
    for (int i = 0; i < 3; ++i)
        o[i * 4 + 3] = -(t[i] * t[3] + t[4 + i] * t[7] + t[8 + i] * t[11]);
    o[12] = T(0); o[13] = T(0); o[14] = T(0); o[15] = T(1);
}

template <typename T> RD void se3_mul(const T* a, const T* b, T* o) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            o[i * 4 + j] = a[i * 4] * b[j] + a[i * 4 + 1] * b[4 + j] + a[i * 4 + 2] * b[8 + j];
        o[i * 4 + 3] = a[i * 4] * b[3] + a[i * 4 + 1] * b[7] + a[i * 4 + 2] * b[11] + a[i * 4 + 3];
    }
    o[12] = T(0); o[13] = T(0); o[14] = T(0); o[15] = T(1);
}

// y = R x (+ p if p != nullptr), R and p read with leading dimension ld
template <typename T> RD void act(const T* r, int ld, const T* p, const T* x, T* y) {
    for (int i = 0; i < 3; ++i) {
        y[i] = r[i * ld] * x[0] + r[i * ld + 1] * x[1] + r[i * ld + 2] * x[2];
        if (p != nullptr) y[i] += p[i * ld];
    }
}

template <typename T> RD bool so3_check(const T* r, int ld, T tol) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (!isfinite(r[i * ld + j])) return false;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            T g = r[i] * r[j] + r[ld + i] * r[ld + j] + r[2 * ld + i] * r[2 * ld + j];
            if (i == j) g -= T(1);
            if (m_fabs(g) > tol) return false;
        }
    T det = r[0] * (r[ld + 1] * r[2 * ld + 2] - r[ld + 2] * r[2 * ld + 1])
          - r[1] * (r[ld] * r[2 * ld + 2] - r[ld + 2] * r[2 * ld])
          + r[2] * (r[ld] * r[2 * ld + 1] - r[ld + 1] * r[2 * ld]);
    return m_fabs(det - T(1)) <= tol;
}

template <typename T> RD bool se3_check(const T* t, T tol) {
    for (int i = 0; i < 16; ++i)
        if (!isfinite(t[i])) return false;
    if (m_fabs(t[12]) > tol || m_fabs(t[13]) > tol || m_fabs(t[14]) > tol) return false;
    if (m_fabs(t[15] - T(1)) > tol) return false;
    return so3_check(t, 4, tol);
}

// o = m @ J(q): rotation about unit axis (1), translation along axis (2), identity (0)
template <typename T> RD void joint_motion(int jtype, const T* axis, T q, const T* m, T* o) {
    if (jtype == 1) {
        T s = m_sin(q), c = m_cos(q), v = T(1) - c;
        T x = axis[0], y = axis[1], z = axis[2];
        T r[9] = {c + x * x * v,     x * y * v - z * s, x * z * v + y * s,
                  y * x * v + z * s, c + y * y * v,     y * z * v - x * s,
                  z * x * v - y * s, z * y * v + x * s, c + z * z * v};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j)
                o[i * 4 + j] = m[i * 4] * r[j] + m[i * 4 + 1] * r[3 + j] + m[i * 4 + 2] * r[6 + j];
            o[i * 4 + 3] = m[i * 4 + 3];
        }
    } else if (jtype == 2) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) o[i * 4 + j] = m[i * 4 + j];
            o[i * 4 + 3] = m[i * 4 + 3]
                         + q * (m[i * 4] * axis[0] + m[i * 4 + 1] * axis[1] + m[i * 4 + 2] * axis[2]);
        }
    } else {
        for (int i = 0; i < 12; ++i) o[i] = m[i];
    }
    o[12] = T(0); o[13] = T(0); o[14] = T(0); o[15] = T(1);
}

template <typename T>
RD void fk_one(int nl, const int* parent, const T* offset, const T* axis, const int* qidx,
               const int* jtype, const T* q, const T* base, T* out) {
    T m[16];
    for (int l = 0; l < nl; ++l) {
        const T* tp = parent[l] < 0 ? base : out + 16 * parent[l];
        se3_mul(tp, offset + 16 * l, m);
        T qv = qidx[l] >= 0 ? q[qidx[l]] : T(0);
        joint_motion(jtype[l], axis + 3 * l, qv, m, out + 16 * l);
    }
}

template <typename T>
__global__ void se3_mul_k(const T* a, const T* b, T* o, long long n, int sa, int sb) {
    GRID_LOOP(i, n) se3_mul(a + 16 * i * sa, b + 16 * i * sb, o + 16 * i);
}

template <typename T> __global__ void se3_inv_k(const T* t, T* o, long long n) {
    GRID_LOOP(i, n) se3_inv(t + 16 * i, o + 16 * i);
}

template <typename T> __global__ void se3_exp_k(const T* xi, T* o, long long n) {
    GRID_LOOP(i, n) se3_exp(xi + 6 * i, o + 16 * i);
}

template <typename T> __global__ void se3_log_k(const T* t, T* o, long long n) {
    GRID_LOOP(i, n) se3_log(t + 16 * i, o + 6 * i);
}

template <typename T>
__global__ void se3_act_batch_k(const T* t, const T* p, T* o, long long n, long long m) {
    GRID_LOOP(k, n * m) {
        long long i = k / m, j = k - i * m;
        act(t + 16 * i, 4, t + 16 * i + 3, p + 3 * j, o + 3 * k);
    }
}

template <typename T> __global__ void se3_act_each_k(const T* t, const T* p, T* o, long long n) {
    GRID_LOOP(i, n) act(t + 16 * i, 4, t + 16 * i + 3, p + 3 * i, o + 3 * i);
}

template <typename T> __global__ void se3_check_k(const T* t, int* bad, long long n, T tol) {
    GRID_LOOP(i, n) if (!se3_check(t + 16 * i, tol)) atomicAdd(bad, 1);
}

template <typename T>
__global__ void fk_tree_k(const int* parent, const T* offset, const T* axis, const int* qidx,
                          const int* jtype, const T* q, const T* base, T* out,
                          long long n, int nl, int nq, int sb) {
    GRID_LOOP(i, n)
        fk_one(nl, parent, offset, axis, qidx, jtype, q + i * nq, base + 16 * i * sb,
               out + 16LL * nl * i);
}

// Trajectory IK (_core_py.ik_dls_traj): one thread per problem, frames in order.
// Poses are kept per path position: the parent of path[i] is path[i - 1].
#define IK_MAXPATH @IKMAXPATH@
#define IK_MAXSEL @IKMAXSEL@
#define IK_MAXQ @IKMAXQ@

// b <- A^-1 b for symmetric positive definite 6x6 A (overwritten)
template <typename T> RD void cholesky_solve6(T* a, T* b) {
    for (int j = 0; j < 6; ++j) {
        T s = a[j * 6 + j];
        for (int k = 0; k < j; ++k) s -= a[j * 6 + k] * a[j * 6 + k];
        s = m_sqrt(s);
        a[j * 6 + j] = s;
        for (int i = j + 1; i < 6; ++i) {
            for (int k = 0; k < j; ++k) a[i * 6 + j] -= a[i * 6 + k] * a[j * 6 + k];
            a[i * 6 + j] /= s;
        }
    }
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < i; ++k) b[i] -= a[i * 6 + k] * b[k];
        b[i] /= a[i * 6 + i];
    }
    for (int i = 5; i >= 0; --i) {
        for (int k = i + 1; k < 6; ++k) b[i] -= a[k * 6 + i] * b[k];
        b[i] /= a[i * 6 + i];
    }
}

// one DLS solve; returns updates applied if converged, -1 otherwise
template <typename T>
RD int ik_dls_one(int npath, const int* path, const T* offset, const T* axis, const int* qidx,
                  const int* jtype, T* q, const T* base, const T* target, int ns, const int* sel,
                  const int* spos, const T* lower, const T* upper, int iters, T tol, T lam2,
                  T* poses, T* jb) {
    T m[16], tinv[16], d[16], xi[6], a[36], w[3], v[3], u[3];
    for (int it = 1; it <= iters; ++it) {
        for (int i = 0; i < npath; ++i) {
            int l = path[i];
            se3_mul(i == 0 ? base : poses + 16 * (i - 1), offset + 16 * l, m);
            joint_motion(jtype[l], axis + 3 * l, qidx[l] >= 0 ? q[qidx[l]] : T(0), m, poses + 16 * i);
        }
        const T* t = poses + 16 * (npath - 1);
        se3_inv(t, tinv);
        se3_mul(tinv, target, d);
        se3_log(d, xi);
        T err = m_sqrt(xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2]
                       + xi[3] * xi[3] + xi[4] * xi[4] + xi[5] * xi[5]);
        if (err < tol) return it - 1;
        for (int s = 0; s < ns; ++s) {     // Ad(T^-1)(w; v) = (R^T w; R^T (v - p x w))
            T* col = jb + 6 * s;
            int i = spos[s];
            if (i < 0) {
                for (int r = 0; r < 6; ++r) col[r] = T(0);
                continue;
            }
            const T* tl = poses + 16 * i;
            const T* ax = axis + 3 * path[i];
            for (int r = 0; r < 3; ++r) u[r] = tl[r * 4] * ax[0] + tl[r * 4 + 1] * ax[1] + tl[r * 4 + 2] * ax[2];
            if (jtype[path[i]] == 1) {
                w[0] = u[0]; w[1] = u[1]; w[2] = u[2];
                v[0] = tl[7] * u[2] - tl[11] * u[1];
                v[1] = tl[11] * u[0] - tl[3] * u[2];
                v[2] = tl[3] * u[1] - tl[7] * u[0];
            } else {
                w[0] = T(0); w[1] = T(0); w[2] = T(0);
                v[0] = u[0]; v[1] = u[1]; v[2] = u[2];
            }
            v[0] -= t[7] * w[2] - t[11] * w[1];
            v[1] -= t[11] * w[0] - t[3] * w[2];
            v[2] -= t[3] * w[1] - t[7] * w[0];
            for (int r = 0; r < 3; ++r) {
                col[r] = t[r] * w[0] + t[4 + r] * w[1] + t[8 + r] * w[2];
                col[3 + r] = t[r] * v[0] + t[4 + r] * v[1] + t[8 + r] * v[2];
            }
        }
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j <= i; ++j) {
                T x = T(0);
                for (int s = 0; s < ns; ++s) x += jb[6 * s + i] * jb[6 * s + j];
                a[i * 6 + j] = x;
                a[j * 6 + i] = x;
            }
            a[i * 6 + i] += lam2;
        }
        cholesky_solve6(a, xi);
        for (int s = 0; s < ns; ++s) {
            T x = T(0);
            for (int i = 0; i < 6; ++i) x += jb[6 * s + i] * xi[i];
            int j = sel[s];
            x += q[j];
            if (x < lower[j]) x = lower[j];
            if (x > upper[j]) x = upper[j];
            q[j] = x;
        }
    }
    return -1;
}

template <typename T>
__global__ void ik_traj_k(const int* paths, const int* npath, int maxd, const T* offset,
                          const T* axis, const int* qidx, const int* jtype, const T* q0, int nq,
                          const T* base, const T* target, int nt, const int* sel, const int* spos,
                          int ns, const T* lower, const T* upper, const int* nframes, int iters,
                          T tol, T lam2, T* out, long long* its, long long* fails, long long np) {
    GRID_LOOP(p, np) {
        T poses[IK_MAXPATH * 16], jb[IK_MAXSEL * 6], q[IK_MAXQ];
        for (int i = 0; i < nq; ++i) q[i] = q0[p * nq + i];
        const int* sp = sel + p * ns;
        long long it_sum = 0, f_sum = 0;
        for (int k = 0; k < nframes[p]; ++k) {
            long long f = p * nt + k;
            int r = ik_dls_one(npath[p], paths + p * maxd, offset, axis, qidx, jtype, q,
                               base + 16 * f, target + 16 * f, ns, sp, spos + p * ns,
                               lower, upper, iters, tol, lam2, poses, jb);
            if (r >= 0) it_sum += r;
            else { it_sum += iters; f_sum += 1; }
            for (int s = 0; s < ns; ++s) out[f * ns + s] = q[sp[s]];
        }
        its[p] = it_sum;
        fails[p] = f_sum;
    }
}

// Monte-Carlo in perturbation form (_core_py.mc_prepare). A "small" transform is
// 12 values, X_R row-major [0..8] and X_p [9..11], standing for I + X.
#define MC_MAXL @MCMAXL@

// o = A^-1 x A for a rigid 4x4 A
template <typename T> RD void sm_conj(const T* x, const T* a, T* o) {
    T t[9], y[3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            t[i * 3 + j] = x[i * 3] * a[j] + x[i * 3 + 1] * a[4 + j] + x[i * 3 + 2] * a[8 + j];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            o[i * 3 + j] = a[i] * t[j] + a[4 + i] * t[3 + j] + a[8 + i] * t[6 + j];
    for (int i = 0; i < 3; ++i)
        y[i] = x[i * 3] * a[3] + x[i * 3 + 1] * a[7] + x[i * 3 + 2] * a[11] + x[9 + i];
    for (int i = 0; i < 3; ++i)
        o[9 + i] = a[i] * y[0] + a[4 + i] * y[1] + a[8 + i] * y[2];
}

// (I + x)(I + y) - I = x + y + x y; o must not alias x or y
template <typename T> RD void sm_compose(const T* x, const T* y, T* o) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            o[i * 3 + j] = x[i * 3 + j] + y[i * 3 + j] + x[i * 3] * y[j]
                         + x[i * 3 + 1] * y[3 + j] + x[i * 3 + 2] * y[6 + j];
        o[9 + i] = x[9 + i] + y[9 + i] + x[i * 3] * y[9] + x[i * 3 + 1] * y[10]
                 + x[i * 3 + 2] * y[11];
    }
}

// J(d) - I: s K + (1 - cos d)(a a^T - I) for revolute, d a for prismatic
template <typename T> RD void sm_joint(int jtype, const T* ax, T d, T* o) {
    for (int i = 0; i < 12; ++i) o[i] = T(0);
    T x = ax[0], y = ax[1], z = ax[2];
    if (jtype == 1) {
        T s = m_sin(d), h = m_sin(T(0.5) * d);
        h = T(2) * h * h;
        o[0] = h * (x * x - T(1)); o[1] = h * x * y - s * z;    o[2] = h * x * z + s * y;
        o[3] = h * y * x + s * z;  o[4] = h * (y * y - T(1));   o[5] = h * y * z - s * x;
        o[6] = h * z * x - s * y;  o[7] = h * z * y + s * x;    o[8] = h * (z * z - T(1));
    } else if (jtype == 2) {
        o[9] = d * x; o[10] = d * y; o[11] = d * z;
    }
}

template <typename T>
__global__ void mc_k(const int* parent, const int* dqidx, const int* jtype, const T* axis,
                     const T* a, const T* nom, const T* coms, const T* w, const T* dq,
                     const T* relnom, int ref, int target, T* pos, T* ang, T* com,
                     long long n, int nl, int nj) {
    GRID_LOOP(s, n) {
        T e[MC_MAXL * 12], c[12], j[12], f[12], y[3];
        const T* d = dq + s * nj;
        T cx = T(0), cy = T(0), cz = T(0);
        for (int l = 0; l < nl; ++l) {
            T* el = e + 12 * l;
            if (parent[l] < 0) {
                for (int i = 0; i < 12; ++i) c[i] = T(0);
            } else {
                sm_conj(e + 12 * parent[l], a + 16 * l, c);
            }
            if (dqidx[l] >= 0) {
                sm_joint(jtype[l], axis + 3 * l, d[dqidx[l]], j);
                sm_compose(c, j, el);
            } else {
                for (int i = 0; i < 12; ++i) el[i] = c[i];
            }
            const T* cl = coms + 3 * l;
            for (int i = 0; i < 3; ++i)
                y[i] = el[i * 3] * cl[0] + el[i * 3 + 1] * cl[1] + el[i * 3 + 2] * cl[2] + el[9 + i];
            const T* nl4 = nom + 16 * l;
            cx += w[l] * (nl4[0] * y[0] + nl4[1] * y[1] + nl4[2] * y[2]);
            cy += w[l] * (nl4[4] * y[0] + nl4[5] * y[1] + nl4[6] * y[2]);
            cz += w[l] * (nl4[8] * y[0] + nl4[9] * y[1] + nl4[10] * y[2]);
        }
        const T* er = e + 12 * ref;   // f = (I + e_ref)^-1 - I
        for (int i = 0; i < 3; ++i) {
            for (int k = 0; k < 3; ++k) f[i * 3 + k] = er[k * 3 + i];
            f[9 + i] = -(er[9 + i] + er[i] * er[9] + er[3 + i] * er[10] + er[6 + i] * er[11]);
        }
        sm_conj(f, relnom, c);
        sm_compose(c, e + 12 * target, j);   // T0^-1 T - I
        for (int i = 0; i < 3; ++i)
            pos[3 * s + i] = relnom[i * 4] * j[9] + relnom[i * 4 + 1] * j[10] + relnom[i * 4 + 2] * j[11];
        y[0] = T(0.5) * (j[7] - j[5]); y[1] = T(0.5) * (j[2] - j[6]); y[2] = T(0.5) * (j[3] - j[1]);
        T sin_t = m_sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
        T cos_t = T(1) + T(0.5) * (j[0] + j[4] + j[8]);
        if (cos_t > T(1)) cos_t = T(1);
        else if (cos_t < T(-1)) cos_t = T(-1);
        ang[s] = m_atan2(sin_t, cos_t);
        com[3 * s] = cx; com[3 * s + 1] = cy; com[3 * s + 2] = cz;
    }
}
"""
_SOURCE = (_SOURCE.replace("@SMALL32@", repr(SMALL_ANGLE["float32"]))
           .replace("@NEARPI32@", repr(NEAR_PI["float32"]))
           .replace("@SMALL64@", repr(SMALL_ANGLE["float64"]))
           .replace("@NEARPI64@", repr(NEAR_PI["float64"]))
           .replace("@MCMAXL@", str(_core_py.MC_MAX_LINKS))
           .replace("@IKMAXPATH@", "32").replace("@IKMAXSEL@", "32").replace("@IKMAXQ@", "64"))
_IK_MAX = {"path": 32, "sel": 32, "q": 64}

_KERNELS = ("se3_mul_k", "se3_inv_k", "se3_exp_k", "se3_log_k", "se3_act_batch_k",
            "se3_act_each_k", "fk_tree_k", "mc_k", "ik_traj_k")
_NAMES = [f"{k}<{c}>" for k in _KERNELS for c in ("float", "double")] + ["se3_check_k<double>"]

_BLOCK = 256
_MAX_BLOCKS = cp.cuda.Device().attributes["MultiProcessorCount"] * 32
_module = None
_dtype = _F32


def _kernel(name, dt):
    global _module
    if _module is None:  # NVRTC compile on first use; CuPy caches the cubin on disk
        _module = cp.RawModule(code=_SOURCE, options=("--std=c++14",), name_expressions=_NAMES)
    return _module.get_function(f"{name}<{_CTYPE[dt]}>")


def _launch(name, dt, n, args, block=_BLOCK):
    blocks = min((n + block - 1) // block, _MAX_BLOCKS)
    _kernel(name, dt)((blocks,), (block,), args)


def set_dtype(dtype):
    """Default compute precision: float32 (default) or float64."""
    global _dtype
    _dtype = _resolve(dtype)


def get_dtype():
    return _dtype


def _resolve(dtype):
    if dtype is None:
        return _dtype
    dt = np.dtype(dtype)
    if dt not in _CTYPE:
        raise ValueError(f"CUDA kernel computes in float32 or float64, not {dt}")
    return dt


def _on_device(*xs):
    return any(isinstance(x, cp.ndarray) for x in xs)


def _dev(x, dt):
    if isinstance(x, cp.ndarray):
        return cp.ascontiguousarray(x, dtype=dt)
    return cp.asarray(np.ascontiguousarray(x, dtype=dt))


def _ret(y, keep):
    return y if keep else cp.asnumpy(y).astype(np.float64, copy=False)


def _empty(shape, dt, keep):
    return cp.zeros(shape, dt) if keep else np.zeros(shape)


def device_name():
    return cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode()


def synchronize():
    cp.cuda.Device().synchronize()


# --------------------------------------------------------------------------- #
# SE(3) batches. Shapes: poses (N, 4, 4), twists (N, 6), points (M, 3).
# --------------------------------------------------------------------------- #
def se3_mul_batch(a, b, dtype=None):
    dt = _resolve(dtype)
    na, nb = a.shape[0], b.shape[0]
    if not (na == nb or na == 1 or nb == 1):
        raise ValueError(f"cannot broadcast batches of length {na} and {nb}")
    keep = _on_device(a, b)
    n = max(na, nb) if min(na, nb) > 0 else 0
    if n == 0:
        return _empty((0, 4, 4), dt, keep)
    out = cp.empty((n, 4, 4), dt)
    _launch("se3_mul_k", dt, n, (_dev(a, dt), _dev(b, dt), out, np.int64(n),
                                 np.int32(na > 1), np.int32(nb > 1)))
    return _ret(out, keep)


def se3_inv_batch(t, dtype=None):
    dt, n, keep = _resolve(dtype), t.shape[0], _on_device(t)
    if n == 0:
        return _empty((0, 4, 4), dt, keep)
    out = cp.empty((n, 4, 4), dt)
    _launch("se3_inv_k", dt, n, (_dev(t, dt), out, np.int64(n)))
    return _ret(out, keep)


def se3_exp_batch(xi, dtype=None):
    dt, n, keep = _resolve(dtype), xi.shape[0], _on_device(xi)
    if n == 0:
        return _empty((0, 4, 4), dt, keep)
    out = cp.empty((n, 4, 4), dt)
    _launch("se3_exp_k", dt, n, (_dev(xi, dt), out, np.int64(n)))
    return _ret(out, keep)


def se3_log_batch(t, dtype=None):
    dt, n, keep = _resolve(dtype), t.shape[0], _on_device(t)
    if n == 0:
        return _empty((0, 6), dt, keep)
    out = cp.empty((n, 6), dt)
    _launch("se3_log_k", dt, n, (_dev(t, dt), out, np.int64(n)))
    return _ret(out, keep)


def se3_act_batch(t, pts, dtype=None):
    """Every pose applied to every point: (N,4,4) x (M,3) -> (N, M, 3)."""
    dt, keep = _resolve(dtype), _on_device(t, pts)
    n, m = t.shape[0], pts.shape[0]
    if n == 0 or m == 0:
        return _empty((n, m, 3), dt, keep)
    out = cp.empty((n, m, 3), dt)
    _launch("se3_act_batch_k", dt, n * m, (_dev(t, dt), _dev(pts, dt), out,
                                           np.int64(n), np.int64(m)))
    return _ret(out, keep)


def se3_act_each(t, pts, dtype=None):
    """i-th pose applied to i-th point: (N,4,4) x (N,3) -> (N, 3)."""
    dt, n, keep = _resolve(dtype), t.shape[0], _on_device(t, pts)
    if pts.shape[0] != n:
        raise ValueError("need exactly one point per pose")
    if n == 0:
        return _empty((0, 3), dt, keep)
    out = cp.empty((n, 3), dt)
    _launch("se3_act_each_k", dt, n, (_dev(t, dt), _dev(pts, dt), out, np.int64(n)))
    return _ret(out, keep)


def se3_check_batch(t, tol, dtype=None):
    """Always float64: a float32 check could not resolve tolerances like 1e-9.
    ``dtype`` is accepted for a uniform signature and ignored."""
    n = t.shape[0]
    if n == 0:
        return True
    bad = cp.zeros(1, cp.int32)
    _launch("se3_check_k", _F64, n, (_dev(t, _F64), bad, np.int64(n), np.float64(tol)))
    return int(bad.get()[0]) == 0


# --------------------------------------------------------------------------- #
# Kinematic tree: one thread per configuration, links in order inside a thread
# --------------------------------------------------------------------------- #
def fk_tree_batch(parent, offset, axis, qidx, jtype, q, base, dtype=None):
    """World poses of all links for N configurations, see ``_core_cy.fk_tree_batch``."""
    dt = _resolve(dtype)
    n, nl, nb = q.shape[0], parent.shape[0], base.shape[0]
    if not (nb == 1 or nb == n):
        raise ValueError("base must have 1 or N poses")
    if q.shape[1] < 1:
        raise ValueError("q needs at least one column (pad with zeros if the tree is fixed)")
    keep = _on_device(q, base)
    if n == 0 or nl == 0:
        return _empty((n, nl, 4, 4), dt, keep)
    out = cp.empty((n, nl, 4, 4), dt)
    i32 = np.dtype(np.int32)
    _launch("fk_tree_k", dt, n, (_dev(parent, i32), _dev(offset, dt), _dev(axis, dt),
                                 _dev(qidx, i32), _dev(jtype, i32), _dev(q, dt), _dev(base, dt),
                                 out, np.int64(n), np.int32(nl), np.int32(q.shape[1]),
                                 np.int32(nb > 1)))
    return _ret(out, keep)


def ik_dls_traj(parent, offset, axis, qidx, jtype, q0, base, link, target, sel, lower, upper,
                n_frames, iters, tol, damping, dtype=np.float64):
    """Trajectory IK, see _core_py.ik_dls_traj (spec): one GPU thread per problem.

    Unlike the other kernels this one defaults to float64: IK residuals feed
    positions that are differentiated twice (ZMP), so float32 cannot reach
    useful tolerances. The work per thread is sequential (warm start), so the
    GPU pays off only with many problems (from ~600 gaits on an RTX 2060 SUPER)."""
    dt = _resolve(dtype)
    sel = np.ascontiguousarray(sel, dtype=np.int32)
    n_p, n_t, ns, nq = base.shape[0], base.shape[1], sel.shape[1], q0.shape[1]
    keep = _on_device(q0, base, target)
    if n_p == 0 or n_t == 0 or ns == 0:
        z = (n_p,)
        return (_empty((n_p, n_t, ns), dt, keep),
                cp.zeros(z, cp.int64) if keep else np.zeros(z, np.int64),
                cp.zeros(z, cp.int64) if keep else np.zeros(z, np.int64))
    link = np.asarray(link)
    paths = [_core_py.tree_path(parent, int(l)) for l in link]
    npath = np.array([len(x) for x in paths], dtype=np.int32)
    if npath.max() > _IK_MAX["path"] or ns > _IK_MAX["sel"] or nq > _IK_MAX["q"]:
        raise ValueError(f"CUDA IK supports paths up to {_IK_MAX['path']} links, "
                         f"{_IK_MAX['sel']} selected and {_IK_MAX['q']} total joints")
    path_a = np.zeros((n_p, npath.max()), dtype=np.int32)
    spos = np.full((n_p, ns), -1, dtype=np.int32)
    for p in range(n_p):
        path_a[p, :npath[p]] = paths[p]
        pos = {int(qidx[l]): i for i, l in enumerate(paths[p]) if qidx[l] >= 0}
        spos[p] = [pos.get(int(s), -1) for s in sel[p]]
    out = cp.zeros((n_p, n_t, ns), dt)                 # frames past n_frames stay 0
    its, fails = cp.empty(n_p, cp.int64), cp.empty(n_p, cp.int64)
    i32 = np.dtype(np.int32)
    # 32 threads per block: few problems spread over many SMs (FP64 units are per SM)
    _launch("ik_traj_k", dt, n_p, (
        _dev(path_a, i32), _dev(npath, i32), np.int32(path_a.shape[1]), _dev(offset, dt),
        _dev(axis, dt), _dev(qidx, i32), _dev(jtype, i32), _dev(q0, dt), np.int32(nq),
        _dev(base, dt), _dev(target, dt), np.int32(n_t), _dev(sel, i32), _dev(spos, i32),
        np.int32(ns), _dev(lower, dt), _dev(upper, dt), _dev(n_frames, i32), np.int32(iters),
        dt.type(tol), dt.type(damping * damping), out, its, fails, np.int64(n_p)), block=32)
    if keep:
        return out, its, fails
    return _ret(out, False), its.get(), fails.get()


def mc_relative_pose(parent, offset, axis, qidx, jtype, q0, base, joints, dq, ref, target,
                     coms, masses, dtype=None):
    """Fused Monte-Carlo, see _core_py.mc_relative_pose (spec) and mc_prepare.
    Only (N,3) + (N,) + (N,3) are written; keep ``dq`` on the device (CuPy) to
    keep the results there as well."""
    dt = _resolve(dtype)
    p = _core_py.mc_prepare(parent, offset, axis, qidx, jtype, q0, base, joints, ref, target,
                            coms, masses)
    keep = _on_device(dq)
    n, nl, nj = dq.shape[0], parent.shape[0], dq.shape[1]
    if n == 0 or nl == 0:
        return _empty((n, 3), dt, keep), _empty((n,), dt, keep), _empty((n, 3), dt, keep)
    pos, ang, com = cp.empty((n, 3), dt), cp.empty(n, dt), cp.empty((n, 3), dt)
    i32 = np.dtype(np.int32)
    _launch("mc_k", dt, n, (_dev(parent, i32), _dev(p["dqidx"], i32), _dev(jtype, i32),
                            _dev(axis, dt), _dev(p["a"], dt), _dev(p["nom"], dt), _dev(coms, dt),
                            _dev(p["w"], dt), _dev(dq, dt) if nj else cp.zeros((n, 1), dt),
                            _dev(p["relnom"], dt), np.int32(ref), np.int32(target),
                            pos, ang, com, np.int64(n), np.int32(nl), np.int32(max(nj, 1))))
    return _ret(pos, keep), _ret(ang, keep), _ret(com, keep)
