"""NDView — borrowed metadata view of a numpy array.

Built ONCE per public call from PythonObject attributes; drivers and kernels
then never touch Python again until return. The data pointer is stored as a
raw address so the struct stays a plain Copyable value type (no origin
parameter); `ptr()` reconstitutes the typed pointer on demand (~free).

Zero-copy contract (SKILL.md §3.1): we read `arr.ctypes.data` directly.
Validation (dtype, ndim bound) happens here, once. Strides are converted to
ELEMENT units (numpy reports bytes).

Rank metadata lives in `DimArray` (`Array[Int, MAX_NDIM]` — Mojo 1.0's
fixed-capacity inline array). Stack storage requires SOME comptime bound;
`MAX_NDIM` is defined once here and shared by every consumer (drivers,
bindings), so the limit is stated in exactly one place. Only the first
`ndim` entries are meaningful; the rest stay zero.
"""

from std.python import PythonObject

comptime MAX_NDIM = 8
"""Maximum supported array rank (numpy's own hard limit is 64; 8 covers all
realistic aggregation use and keeps the view's metadata in ~2 cache lines)."""

comptime DimArray = Array[Int, MAX_NDIM]
"""Fixed-capacity inline array for per-dim metadata (sizes/strides/axes)."""


def _dtype_name[dtype: DType]() -> String:
    comptime if dtype == DType.float64:
        return "float64"
    elif dtype == DType.float32:
        return "float32"
    elif dtype == DType.int64:
        return "int64"
    elif dtype == DType.int32:
        return "int32"
    elif dtype == DType.bool:
        return "bool"
    else:
        return "unsupported"


struct NDView[dtype: DType](Copyable):
    """Shape/strides/address of a borrowed numpy array of `dtype`."""

    var addr: Int
    var ndim: Int
    var sizes: DimArray  # element counts per dim; [ndim, MAX_NDIM) unused
    var strides: DimArray  # ELEMENT strides per dim; [ndim, MAX_NDIM) unused

    def __init__(
        out self,
        addr: Int,
        ndim: Int,
        sizes: DimArray,
        strides: DimArray,
    ):
        self.addr = addr
        self.ndim = ndim
        self.sizes = sizes.copy()
        self.strides = strides.copy()

    @staticmethod
    def from_numpy(arr: PythonObject, op: String) raises -> Self:
        comptime want = _dtype_name[Self.dtype]()
        var got = String(py=arr.dtype.name)
        if got != want:
            raise Error(op + ": expected dtype " + want + ", got " + got)
        var ndim = Int(py=arr.ndim)
        if ndim > MAX_NDIM:
            raise Error(
                op
                + ": ndim "
                + String(ndim)
                + " > MAX_NDIM="
                + String(MAX_NDIM)
                + " not supported"
            )
        var itemsize = Int(py=arr.dtype.itemsize)
        var sizes = DimArray(fill=0)
        var strides = DimArray(fill=0)
        for d in range(ndim):
            sizes[d] = Int(py=arr.shape[d])
            strides[d] = Int(py=arr.strides[d]) // itemsize
        return Self(Int(py=arr.ctypes.data), ndim, sizes, strides)

    def ptr(self) -> Pointer[mut=True, Scalar[Self.dtype], MutAnyOrigin]:
        return Pointer[mut=True, Scalar[Self.dtype], MutAnyOrigin](
            unsafe_from_address=self.addr
        )

    def numel(self) -> Int:
        var total = 1
        for d in range(self.ndim):
            total *= self.sizes[d]
        return total
