"""Cost model: fuse smooth+NVFP4 quantize into the GEMM producer, LoRA-down stays a separate
kernel (reads x, writes down). Per-element quantize cost and the measured prefix pieces are
from this card; L2 bandwidth is inferred (see prior model)."""
P4, P16, BW = 873e12, 288e12, 1111e9
L2BW = 6.5e12            # inferred in the earlier model from an ncu % of peak; not directly measured
INST = 10.33             # thread-instr / element, standalone smooth quantize (ncu, measured)
SMSP_ISSUE = 110 * 4 * 2.35e9   # warp-instr/s, one issue per SMSP per cycle at ~2.35 GHz under the power cap
r = 128
shapes = [(m, n, k) for (n, k) in ((3072, 3072), (12288, 3072), (3072, 12288)) for m in (4096, 6889, 9216, 16384)]
print("| shape | serial SOL now | GEMM SOL | quant ALU +us (TN128 / 256) | A L2 bytes GB now -> fused | L2 %peak at GEMM SOL | down-only prefix SOL | partial-fusion SOL (best TN) | gain vs serial SOL |")
print("|---|---|---|---|---|---|---|---|---|")
for m, n, k in shapes:
    fp4 = 2*m*n*k/P4; up = 2*m*n*r/P16; dn = 2*m*k*r/P16
    g_b = m*k*0.5625 + n*k*0.5625 + m*r*2 + n*r*2 + n*2 + m*n*2
    gemm = max(fp4 + up, g_b/BW)
    pre_b = m*k*2 + m*k*0.5625 + m*r*2 + k*r*2
    pre = max(pre_b/BW, dn)
    serial = pre + gemm
    # partial: prefix reads x + L2T, writes down only
    pre2 = max((m*k*2 + m*r*2 + k*r*2)/BW, dn)
    best = None; cells = []
    for tn in (128, 256):
        rep = n / tn
        quant_us = m*k*rep*INST/32 / SMSP_ISSUE      # extra issue, if not hidden behind MMA
        a_l2 = m*k*2*rep                              # bf16 x re-read per N tile from L2
        b_l2 = n*k*0.5625*(m/128)
        l2_t = (a_l2 + b_l2 + m*n*2) / L2BW
        # GEMM-side DRAM: x replaces xq/sf
        g2_b = m*k*2 + n*k*0.5625 + m*r*2 + n*r*2 + n*2 + m*n*2
        # optimistic: quantize issue overlaps MMA; pessimistic: adds
        g_opt = max(fp4 + up, g2_b/BW, l2_t)
        g_pes = max(fp4 + up + quant_us, g2_b/BW, l2_t)
        cells.append((tn, quant_us, a_l2, l2_t, g_opt, g_pes))
        cand = (pre2 + g_opt, pre2 + g_pes, tn)
        if best is None or cand[0] < best[0]: best = cand
    a_now = m*k*0.5625*(n/128)
    c128, c256 = cells
    print(f"| {m}x{n}x{k} | {serial*1e6:.0f} | {gemm*1e6:.0f} | +{c128[1]*1e6:.0f} / +{c256[1]*1e6:.0f} | {a_now/1e9:.1f} -> {c128[2]/1e9:.1f} | "
          f"{100*c128[3]/gemm:.0f}% / {100*c256[3]/gemm:.0f}% | {pre2*1e6:.0f} | {best[0]*1e6:.0f}-{best[1]*1e6:.0f} (TN{best[2]}) | "
          f"{100*(1-best[0]/serial):.0f}% to {100*(1-best[1]/serial):.0f}% |")
