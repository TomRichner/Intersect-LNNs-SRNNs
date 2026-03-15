julia> empty!(ARGS); append!(ARGS, ["--epochs", "10", "--size", "32", "--bs", "256", "--lr", "0.01"])
8-element Vector{String}:
 "--epochs"
 "10"
 "--size"
 "32"
 "--bs"
 "256"
 "--lr"
 "0.01"

julia> include("JuliaLang/scripts/train_har_srnn.jl")
HAR Training — SRNNCell (semi_implicit, batched BPTT)
  Model size: 32, n_E: 16, n_I: 16
  per_neuron: false
  SFA timescales (n_a_E): 3, STD (n_b_E): 0
  Solver: semi_implicit, h: 0.02, unfolds: 6
  Readout: synaptic
  LR: 0.01, Epochs: 10, Batch: 256
Loading HAR data from: /Users/tom/Desktop/local_code/Intersect-LNNs-SRNNs/JuliaLang/scripts/../data/har/UCI HAR Dataset
  Raw train: 7352 samples × 561 features
  Raw test:  2947 samples × 561 features
  Total training sequences: 7336
  Total test sequences:     367
  Validation split: 733, training split: 6603
  Cell params: 18982
  Head params: 198
  Total params: 19180
  State dim: 80

Gradient smoke test (batched)...
  Initial loss: 2.0083 (expected ~1.79 = -log(1/6))
  All SRNN gradients present ✓
  Head weight gradient norm: 5.901922
  Head bias gradient norm: 0.37425438

Starting training...

Epochs 000, train loss: 1.61, train accuracy: 36.58%, valid loss: 2.00, valid accuracy: 31.79%, test loss: 2.03, test accuracy: 28.88%
Epochs 001, train loss: 1.32, train accuracy: 44.81%, valid loss: 1.47, valid accuracy: 32.20%, test loss: 1.48, test accuracy: 34.33%
Epochs 002, train loss: 0.99, train accuracy: 56.75%, valid loss: 1.16, valid accuracy: 42.43%, test loss: 1.14, test accuracy: 48.23%
Epochs 003, train loss: 0.79, train accuracy: 68.05%, valid loss: 0.89, valid accuracy: 54.02%, test loss: 0.86, test accuracy: 62.13%
Epochs 004, train loss: 0.65, train accuracy: 79.27%, valid loss: 0.74, valid accuracy: 78.58%, test loss: 0.72, test accuracy: 75.75%
Epochs 005, train loss: 0.55, train accuracy: 81.11%, valid loss: 0.62, valid accuracy: 81.58%, test loss: 0.63, test accuracy: 77.93%
Epochs 006, train loss: 0.49, train accuracy: 81.83%, valid loss: 0.56, valid accuracy: 76.13%, test loss: 0.57, test accuracy: 78.47%
Epochs 007, train loss: 0.43, train accuracy: 87.08%, valid loss: 0.48, valid accuracy: 86.36%, test loss: 0.50, test accuracy: 85.83%
Epochs 008, train loss: 0.36, train accuracy: 93.16%, valid loss: 0.42, valid accuracy: 90.18%, test loss: 0.47, test accuracy: 85.01%
Epochs 009, train loss: 0.31, train accuracy: 94.39%, valid loss: 0.34, valid accuracy: 94.82%, test loss: 0.39, test accuracy: 90.74%
Best epoch 009, train loss: 0.31, train accuracy: 94.39%, valid loss: 0.34, valid accuracy: 94.82%, test loss: 0.39, test accuracy: 90.74%
(cell = (a_0 = Float32[0.42624137], W_in = Float32[-0.115264036 0.23902255 … -0.15374148 -0.0040708696; 0.22856359 0.0383474 … 0.10781211 0.10842775; … ; -0.060691137 -0.17125873 … 0.21753101 -0.058993228; 0.07355944 -0.113456406 … 0.11705077 -0.2603248], log_tau_a_E_lo = Float32[-1.2318527], log_tau_a_E_hi = Float32[10.197442], W = Float32[0.22399329 0.114662886 … -0.050451316 -0.51924455; 0.05796093 0.104269326 … -0.1694432 0.32967073; … ; 0.016449139 0.6169197 … 0.4980827 0.26021117; 0.4267647 -0.554343 … -0.26900774 -0.032404955], log_c_E = Float32[-3.227613], c_0_E = Float32[0.07818663], log_tau_d = Float32[-2.7570739]), head = (weight = Float32[-0.26730993 -0.002419514 … -1.1463783 -0.007886867; -0.4815608 0.37758183 … 1.2769156 0.16693996; … ; -0.32046098 0.27355802 … -1.0337882 0.0049115806; -0.21636862 -0.24068435 … -0.39744553 0.09981254], bias = Float32[0.036674596, -0.023429094, -0.24274388, -0.062243674, 0.09160099, 0.06680046]))