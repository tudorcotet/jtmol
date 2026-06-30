## Size sweep (NVIDIA A100-SXM4-40GB, K=128)

| Label | N atoms | chunk | jtmol E | upstream E | speedup E | jtmol grad | upstream grad | speedup grad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 75res_2mtq | 1116 | 4 | 0.722 | 27.797 | 38.5× | 4.491 | 30.833 | 6.9× |
| 100res_5umr | 1591 | 2 | 1.634 | 27.315 | 16.7× | 10.335 | 31.270 | 3.0× |
| 150res_5yzf | 2465 | 1 | 6.113 | 26.599 | 4.4× | 34.230 | 31.235 | 0.9× |
| 200res_5mm8 | 3039 | 1 | 9.276 | 27.980 | 3.0× | 56.691 | 33.087 | 0.6× |
| 300res_6f8b | 4466 | 1 | 17.372 | 26.841 | 1.5× | 156.404 | 30.883 | 0.2× |

Errors:
- 400res_6azu: JaxRuntimeError('INTERNAL: Autotuning failed for HLO: %loop_negate_fusion = f64[1,1,5899,5899,16]{4,3,2,1,0} fusion(%get-tuple-element.668, %loop_select_fusion, %input_transpose_fusion.5, %get-tuple-element.655, %get-tuple-element.654), kind=kLoop, calls=%fused_negate, metadata={op_name="jit(fn)/while/body/closed_call/vmap(transpose(jvp()))/cond/branch_1_fun/vmap(vmap(vmap()))/neg" stack_frame_id=5816}. Failed to pick best config: All configs failed during profiling.\nFailures (2):\nEXECUTION FAILED: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 4.15GiB. [tf-allocator-allocation-error=\'\']\nEXECUTION FAILED: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 4.15GiB. [tf-allocator-allocation-error=\'\']')
- 500res_5v2d: JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 9.16GiB. [tf-allocator-allocation-error=''] [executable_name='jit_fn']")
- 600res_5m4a: JaxRuntimeError('RESOURCE_EXHAUSTED: Autotuning failed for HLO: %input_reduce_fusion.22 = f64[8800,8800]{0,1} fusion(%loop_select_fusion.2), kind=kInput, calls=%fused_reduce.29, metadata={op_name="jit(fn)/while/body/closed_call/vmap()/cond/branch_1_fun/vmap(vmap(vmap()))/reduce_max" stack_frame_id=4534}. Failed to profile configs: Out of memory while trying to allocate 9.25GiB. [tf-allocator-allocation-error=\'\']')