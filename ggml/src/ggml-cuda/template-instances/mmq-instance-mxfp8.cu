// ROC8: MXFP8 mmq instantiation. Hand-written (not generate_cu_files.py --
// that script's own header says "do not edit manually" and regenerating it
// would touch every sibling file across two concurrent branches); mechanical
// copy of mmq-instance-f8e4m3.cu/mmq-instance-f8e5m2.cu's one-line pattern.

#include "../mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_MXFP8);
