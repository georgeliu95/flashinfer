"""Direction B cost model: full fusion (bf16 x -> smooth+NVFP4 quant + LoRA-down inside the GEMM)."""
P4, P16, BW = 873e12, 288e12, 1111e9
L2BW = 6.5e12          # L2 peak implied by big.ncu-rep (r128 GEMM): 100.7 Gsector/s x 32 B = 3.22 TB/s at 49.5% of peak; L2 = 96 MiB
SMEM = 100 * 1024      # usable dyn smem / CTA (opt-in 101376 B)
r = 128
shapes = [(m, n, k) for (n, k) in ((3072, 3072), (12288, 3072), (3072, 12288)) for m in (4096, 6889, 9216, 16384)]
print("shape | fp4 us | up us | down us | fusedSOL | overlapSOL | split-serial SOL | down x(N/TN) TN=128/256 extra us | A-side L2->SM GB TN128/256 (fp4 today) | L2 %peak @fusedSOL TN128/256 | DRAM bytes MB fused/overlap")
for m, n, k in shapes:
    fp4 = 2*m*n*k/P4; up = 2*m*n*r/P16; dn = 2*m*k*r/P16
    comp = fp4 + up + dn
    b_f = m*k*2 + k*r*2 + n*k*0.5625 + n*r*2 + n*2 + m*n*2
    inter = 2*(m*k*0.5625 + m*r*2)                 # xq+sf+down written once and read once
    b_o = b_f + inter
    fused = max(comp, b_f/BW); over = max(comp, b_o/BW)
    pre = max((m*k*2 + m*k*0.5625 + m*r*2 + k*r*2)/BW, dn)
    b_g = m*k*0.5625 + n*k*0.5625 + m*r*2 + n*r*2 + n*2 + m*n*2
    serial = pre + max(fp4+up, b_g/BW)
    out = []
    for tn in (128, 256):
        rep = n/tn
        extra_dn = dn*(rep-1)
        a_bytes = m*k*2*rep
        b_bytes = n*k*0.5625*(m/128)
        out.append((rep, extra_dn, a_bytes, b_bytes))
    a_today = m*k*0.5625*(n/128)
    l2 = [(o[2]+o[3]+m*n*2)/fused/L2BW*100 for o in out]
    print(f"{m}x{n}x{k} | {fp4*1e6:.0f} | {up*1e6:.0f} | {dn*1e6:.0f} | {fused*1e6:.0f} | {over*1e6:.0f} | {serial*1e6:.0f} | "
          f"x{out[0][0]:.0f}: +{out[0][1]*1e6:.0f} / x{out[1][0]:.0f}: +{out[1][1]*1e6:.0f} | "
          f"{out[0][2]/1e9:.2f}/{out[1][2]/1e9:.2f} ({a_today/1e9:.2f}) | {l2[0]:.0f}%/{l2[1]:.0f}% | {b_f/1e6:.0f}/{b_o/1e6:.0f} (DRAM-time {b_f/BW*1e6:.0f}/{b_o/BW*1e6:.0f} us)")
# smem per stage for producer-side quantize, K-tile 128
print("\nsmem/stage (K tile 128): bf16 x tile + fp4 A + SFA + fp4 B + SFB")
for tm, tn in ((128, 128), (256, 128), (128, 256)):
    x = tm*128*2; a = tm*128//2; sfa = tm*128//16; b = tn*128//2; sfb = tn*128//16
    l2t = 128*r*2  # LoRA-down weight slice L2T[k-tile, rank] bf16, needed for in-CTA down
    tot = x + a + sfa + b + sfb
    print(f"  {tm}x{tn}: x {x//1024}K + A {a//1024}K + SFA {sfa//1024}K + B {b//1024}K + SFB {sfb//1024}K = {tot/1024:.0f}K -> {SMEM//tot} stage(s); +L2T {l2t//1024}K = {(tot+l2t)/1024:.0f}K -> {SMEM//(tot+l2t)} stage(s) in 100K (today fp4-only {(a+sfa+b+sfb)/1024:.0f}K/stage)")
# CTA-owns-M-stripe variant: quantized A stripe resident in smem?
print("\nM-stripe-resident quantized A (TileM x K x 0.5625 B):")
for tm in (128, 256):
    for k in (3072, 12288):
        print(f"  TileM {tm}, K {k}: {tm*k*0.5625/1024:.0f} KB (smem 100 KB); stripes at M=16384: {16384//tm} for 110 SMs")
# producer quantize ALU: ASSUMED ~4.4 thread-instr/elem (smooth mul, amax, scale, cvt e2m1x2, pack); estimate, not measured
print("\nper-N-tile quantize ALU (4.4 thr-inst/elem) as % of issue over fused SOL, M=16384:")
for n, k in ((3072, 3072), (12288, 3072), (3072, 12288)):
    m = 16384
    comp = 2*m*n*k/P4 + 2*m*n*r/P16 + 2*m*k*r/P16
    for tn in (128, 256):
        warp_inst = m*k*(n/tn)*4.4/32
        print(f"  {n}x{k} TN {tn}: {warp_inst/1e6:.0f}M warp-inst = {warp_inst/(110*4*2.35e9)*1e6:.0f} us of all-SMSP issue ({100*warp_inst/(110*4*2.35e9)/comp:.0f}% of SOL window)")
