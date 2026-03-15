julia> empty!(ARGS); append!(ARGS, ["--epochs", "10", "--size", "32", "--bs", "256", "--lr", "0.02", "--per_neuron"])
9-element Vector{String}:
 "--epochs"
 "10"
 "--size"
 "32"
 "--bs"
 "256"
 "--lr"
 "0.02"
 "--per_neuron"

julia> include("JuliaLang/scripts/train_har_srnn.jl")

HAR Training — SRNNCell (semi_implicit, batched BPTT)
  Model size: 32, n_E: 16, n_I: 16
  per_neuron: true
  SFA timescales (n_a_E): 3, STD (n_b_E): 0
  Solver: semi_implicit, h: 0.02, unfolds: 6
  Readout: synaptic
  LR: 0.02, Epochs: 10, Batch: 256
Loading HAR data from: /Users/tom/Desktop/local_code/Intersect-LNNs-SRNNs/JuliaLang/scripts/../data/har/UCI HAR Dataset
  Raw train: 7352 samples × 561 features
  Raw test:  2947 samples × 561 features
  Total training sequences: 7336
  Total test sequences:     367
  Validation split: 733, training split: 6603
  Cell params: 19104
  Head params: 198
  Total params: 19302
  State dim: 80

Gradient smoke test (batched)...
  Initial loss: 2.0084 (expected ~1.79 = -log(1/6))
  All SRNN gradients present ✓
  Head weight gradient norm: 5.901979
  Head bias gradient norm: 0.37425223

Starting training...

Epochs 000, train loss: 1.67, train accuracy: 30.02%, valid loss: 2.00, valid accuracy: 31.79%, test loss: 2.03, test accuracy: 28.88%
Epochs 001, train loss: 1.36, train accuracy: 37.91%, valid loss: 1.48, valid accuracy: 33.56%, test loss: 1.50, test accuracy: 32.15%
Epochs 002, train loss: 1.15, train accuracy: 42.14%, valid loss: 1.23, valid accuracy: 40.93%, test loss: 1.23, test accuracy: 40.60%
Epochs 003, train loss: 1.01, train accuracy: 53.91%, valid loss: 1.08, valid accuracy: 44.20%, test loss: 1.07, test accuracy: 42.78%
Epochs 004, train loss: 0.85, train accuracy: 60.48%, valid loss: 0.93, valid accuracy: 57.03%, test loss: 0.92, test accuracy: 58.58%
Epochs 005, train loss: 0.76, train accuracy: 62.52%, valid loss: 0.83, valid accuracy: 59.07%, test loss: 0.83, test accuracy: 60.22%
Epochs 006, train loss: 0.70, train accuracy: 62.98%, valid loss: 0.78, valid accuracy: 58.94%, test loss: 0.76, test accuracy: 59.95%
Epochs 007, train loss: 0.68, train accuracy: 65.44%, valid loss: 0.72, valid accuracy: 63.44%, test loss: 0.71, test accuracy: 62.13%
Epochs 008, train loss: 0.66, train accuracy: 65.16%, valid loss: 0.71, valid accuracy: 61.66%, test loss: 0.73, test accuracy: 61.85%
Epochs 009, train loss: 0.62, train accuracy: 70.81%, valid loss: 0.69, valid accuracy: 69.71%, test loss: 0.67, test accuracy: 71.66%
Best epoch 009, train loss: 0.62, train accuracy: 70.81%, valid loss: 0.69, valid accuracy: 69.71%, test loss: 0.67, test accuracy: 71.66%
(cell = (a_0 = Float32[0.21714932, 0.44365203, 0.4700291, 0.47019044, 0.4700475, 0.4706343, 0.47010827, 0.47023013, 0.21995154, 0.48070842  …  0.22025937, 0.17637052, 0.24376908, 0.4048383, 0.46319956, 0.4651835, 0.1883806, 0.21526584, 0.4656021, 0.22992417], W_in = Float32[0.043428194 0.2667749 … -0.19999492 0.086195186; 0.09679244 0.119855046 … 0.0018610143 0.0064180903; … ; -0.10586832 -0.14422339 … -0.0148861315 0.14378074; 0.18004708 -0.2266367 … 0.17790568 -0.077450745], log_tau_a_E_lo = Float32[-1.1535041, -1.2592273, -1.1523019, -1.1388295, -1.3779778, -1.3769704, -1.3775934, -1.1391414, -1.3752366, -1.7570661, -1.2386757, -1.1087233, -1.37921, 0.90836066, 0.32987505, -1.5094303], log_tau_a_E_hi = Float32[9.895354, 9.99995, 9.881141, 9.880357, 9.879885, 9.882039, 9.879906, 9.879991, 10.12897, 9.868082, 9.892435, 9.482318, 9.879577, 9.949452, 9.939398, 9.873571], W = Float32[0.48981503 0.08276192 … 0.11050091 -0.3714963; -0.059665598 0.016407045 … -0.24531473 0.25446782; … ; 0.35880977 0.50633717 … -0.09866095 0.31384498; 0.43907326 -0.4389093 … -0.12564003 0.2535358], log_c_E = Float32[-2.8956914, -2.999328, -2.8799298, -2.8799083, -2.8798985, -2.8798904, -2.8799706, -2.8798952, -3.1296756, -2.8473864, -2.9012723, -2.5996692, -2.8794065, -3.2735431, -3.083671, -2.868766], c_0_E = Float32[-0.11624804, 0.0023005174, 0.12012446, 0.12010062, 0.1200724, 0.12107689, 0.12009712, 0.12010672, -0.12934321, 0.10860989, 0.09690723, 0.054608624, 0.1211185, -0.003978368, -0.10090825, 0.13079824], log_tau_d = Float32[-2.3610568, -2.2908792, -2.1304836, -2.132065, -2.3810337, -2.370429, -2.1054232, -2.3722732, -2.2372959, -3.2738745  …  -2.35642, -2.0752244, -1.9059565, -3.0950925, -2.2900276, -2.0298672, -2.214358, -2.4354522, -2.127879, -2.3355343]), head = (weight = Float32[-0.13556731 0.027112387 … -0.032442532 -0.13094677; -0.26994044 0.29104716 … 0.13071112 0.18653266; … ; -0.14135899 0.14801571 … -0.06947539 -0.05367298; -0.16745912 -0.22607863 … 0.05699471 0.046073668], bias = Float32[-0.0027798628, -0.06573074, 0.11511313, 0.019196168, 0.0035368837, 0.02602534]))