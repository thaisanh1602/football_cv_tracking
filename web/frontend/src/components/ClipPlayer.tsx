import { useEffect, useRef } from "react";
import { drawOverlay, drawRadar, radarHeight } from "../draw";
import type { Source } from "../source";
import type { ClipInfo, ClipMeta, FrameAnn, ViewOptions } from "../types";
import { useAnnotationStream } from "../useAnnotationStream";

const RADAR_W = 520;
const TRAIL_FRAMES = 25; // vệt 1 s ở 25 fps

export function ClipPlayer({ clip, meta, opt, source }: { clip: ClipInfo; meta: ClipMeta; opt: ViewOptions; source: Source }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const radarRef = useRef<HTMLCanvasElement>(null);
  const statusRef = useRef<HTMLDivElement>(null);
  const { connected, need, get } = useAnnotationStream(clip.id, meta.video.frames, source.allFrames(clip.id));

  useEffect(() => {
    const video = videoRef.current!;
    const octx = overlayRef.current!.getContext("2d")!;
    const rctx = radarRef.current!.getContext("2d")!;
    const { fps, frames } = meta.video;
    // mediaTime của frame đang HIỂN THỊ (requestVideoFrameCallback), chính xác hơn currentTime khi đang phát
    // Firefox cũ chưa có requestVideoFrameCallback -> dùng currentTime
    const hasRVFC = "requestVideoFrameCallback" in HTMLVideoElement.prototype;
    let mediaTime = video.currentTime;
    let rvfc = 0;
    let raf = 0;

    const onFrame: VideoFrameRequestCallback = (_, m) => {
      mediaTime = m.mediaTime;
      rvfc = video.requestVideoFrameCallback(onFrame);
    };
    const sync = () => (mediaTime = video.currentTime);
    if (hasRVFC) rvfc = video.requestVideoFrameCallback(onFrame);
    video.addEventListener("seeked", sync);
    video.addEventListener("pause", sync);

    const loop = () => {
      const t = hasRVFC && !video.paused ? mediaTime : video.currentTime;
      const f = Math.min(Math.max(Math.round(t * fps), 0), frames - 1);
      need(f);
      const fr = get(f);
      drawOverlay(octx, fr, meta, opt);
      const history: FrameAnn[] = [];
      for (let k = Math.max(f - TRAIL_FRAMES, 0); k <= f; k++) {
        const h = get(k);
        if (h) history.push(h);
      }
      drawRadar(rctx, fr, history, meta, opt);
      if (statusRef.current) {
        statusRef.current.textContent = fr
          ? `frame ${f + 1}/${frames} · ${fr.p.length} người · radar ${fr.h ? "có" : "không có"} homography`
          : `frame ${f + 1}/${frames} · đang tải annotation…`;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      if (rvfc) video.cancelVideoFrameCallback(rvfc);
      video.removeEventListener("seeked", sync);
      video.removeEventListener("pause", sync);
    };
  }, [meta, opt, need, get]);

  return (
    <div className="player">
      <div className="video-wrap" style={{ aspectRatio: `${meta.video.width} / ${meta.video.height}` }}>
        <video ref={videoRef} src={source.videoUrl(clip.id)} controls muted playsInline loop preload="auto" />
        <canvas ref={overlayRef} width={meta.video.width} height={meta.video.height} className="overlay" />
      </div>
      <div className="below">
        <canvas ref={radarRef} width={RADAR_W} height={radarHeight(meta, RADAR_W)} className="radar" />
        <div className="status">
          <span className={`dot ${connected ? "on" : ""}`} />{" "}
          {source.mode === "static" ? "Annotation tải từ file tĩnh (bản demo không có backend)" : connected ? "WebSocket đã kết nối" : "Đang kết nối…"}
          <div ref={statusRef} className="muted" />
        </div>
      </div>
    </div>
  );
}
