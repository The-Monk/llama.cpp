import sys
sys.path.insert(0, "/home/jmonk/roc10/gguf-py")
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

SRC_BODY = sys.argv[1]   # Bonsai ternary target
SRC_MTP  = "/aipool/models/qwen3.6-27b-bf16-mtp/Qwen3.6-27B-F8E4M3-nextnBF16.gguf"
DST      = sys.argv[2]

body = GGUFReader(SRC_BODY)
mtp  = GGUFReader(SRC_MTP)

body_names = {t.name for t in body.tensors}
graft = [t for t in mtp.tensors if t.name not in body_names]
print(f"grafting {len(graft)} tensors from MTP file:")
for t in graft:
    print(f"  {t.name:52s} {str(t.tensor_type)[-10:]:10s} {list(t.shape)}")
assert all(t.name.startswith("blk.64.") for t in graft), "unexpected non-blk.64 graft tensor"

w = GGUFWriter(DST, "qwen35")

overrides = {
    "qwen35.block_count": (65, GGUFValueType.UINT32),
    "qwen35.nextn_predict_layers": (1, GGUFValueType.UINT32),
}
# copy the nextn KV's true type from the MTP file if present
f = mtp.fields.get("qwen35.nextn_predict_layers")
if f:
    overrides["qwen35.nextn_predict_layers"] = (f.contents(), f.types[0])

seen = set()
for field in body.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
    if field.name in overrides:
        v, vt = overrides[field.name]
        w.add_key_value(field.name, v, vt)
        seen.add(field.name)
        print(f"override {field.name} -> {v}")
    else:
        w.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
for k, (v, vt) in overrides.items():
    if k not in seen:
        w.add_key_value(k, v, vt)
        print(f"add {k} = {v}")

total = 0
for t in list(body.tensors) + graft:
    total += t.n_bytes
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()

done = 0
for t in list(body.tensors) + graft:
    src = body if t.name in body_names else mtp
    w.write_tensor_data(t.data, tensor_endianess=src.endianess)
    done += t.n_bytes
    if done % (1 << 30) < t.n_bytes:
        print(f"  {done/2**30:.1f} GiB written")
w.close()
print(f"DONE: {DST} ({done/2**30:.2f} GiB, {len(body.tensors)+len(graft)} tensors)")
