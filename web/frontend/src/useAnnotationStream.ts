import { useCallback, useEffect, useRef, useState } from "react";
import type { FrameAnn } from "./types";

const CHUNK = 100; // số frame mỗi lần xin (~4 s ở 25 fps)
const PREFETCH = 50; // còn ít hơn ngần này frame phía trước thì xin tiếp

/**
 * Bộ đệm annotation nhận qua WebSocket. Gọi need(frame) mỗi khi vẽ: nếu phía trước frame đó chưa có dữ liệu
 * (hoặc vừa tua tới chỗ chưa có), xin thêm 1 khoảng. get(frame) trả annotation nếu đã có.
 */
export function useAnnotationStream(clipId: string | null, totalFrames: number) {
  const buffer = useRef(new Map<number, FrameAnn>());
  const pending = useRef(new Map<number, number>()); // các khoảng [from, to) đang chờ trả lời
  const ws = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    buffer.current.clear();
    pending.current.clear();
    if (!clipId) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${proto}://${location.host}/ws/clips/${clipId}`);
    ws.current = socket;
    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);
    socket.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type !== "frames") return;
      for (const fr of msg.frames as FrameAnn[]) buffer.current.set(fr.f, fr);
      pending.current.delete(msg.from);
    };
    return () => socket.close();
  }, [clipId]);

  const isPending = (f: number) => {
    for (const [a, b] of pending.current) if (f >= a && f < b) return true;
    return false;
  };

  const need = useCallback((frame: number) => {
    const socket = ws.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    for (let f = frame; f < Math.min(frame + PREFETCH, totalFrames); f++) {
      if (buffer.current.has(f) || isPending(f)) continue;
      const to = Math.min(f + CHUNK, totalFrames);
      pending.current.set(f, to);
      socket.send(JSON.stringify({ type: "range", from: f, to }));
      return;
    }
  }, [totalFrames]);

  const get = useCallback((frame: number) => buffer.current.get(frame), []);

  return { connected, need, get };
}
