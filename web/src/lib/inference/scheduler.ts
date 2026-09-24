/** A reset invalidates pending work but cannot unlock a newer session's frame. */
export class FrameGate {
  private session = 0;
  private frame = 0;
  private active: number | null = null;
  begin(): { sessionId: number; frameId: number } | null {
    if (this.active !== null) return null;
    const frameId = ++this.frame;
    this.active = frameId;
    return { sessionId: this.session, frameId };
  }
  finish(sessionId: number, frameId: number): boolean {
    if (sessionId !== this.session || frameId !== this.active) return false;
    this.active = null;
    return true;
  }
  isCurrent(sessionId: number): boolean { return sessionId === this.session; }
  reset(): number { this.session += 1; this.active = null; return this.session; }
}
