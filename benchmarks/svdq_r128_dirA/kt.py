import sqlite3,sys
c=sqlite3.connect(sys.argv[1]); n=int(sys.argv[2]) if len(sys.argv)>2 else 40
rows=c.execute("""select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory,k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start""").fetchall()
prev=None
for r in rows[-n:]:
    gap=(r[0]-prev)/1e3 if prev else 0
    print(f"{(r[1]-r[0])/1e3:8.1f}us gap{gap:7.1f} st{r[10]} {r[2][:40]:40s} grid={r[3]}x{r[4]}x{r[5]} blk={r[6]} regs={r[7]} smem={r[8]}+{r[9]}")
    prev=r[1]
