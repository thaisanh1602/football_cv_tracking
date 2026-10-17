import { useEffect, useState } from "react";
import { ClipPlayer } from "./components/ClipPlayer";
import { UploadPanel } from "./components/UploadPanel";
import type { ClipInfo, ClipMeta, ViewOptions } from "./types";

const REPO = "https://github.com/thaisanh1602/football_cv_tracking";

const METRICS = [
  { value: "53.6", label: "HOTA tracking", note: "ID switch −72%" },
  { value: "0.94", label: "Accuracy phân đội", note: "SigLIP + UMAP" },
  { value: "1.27 m", label: "Sai số radar", note: "trung vị, SoccerNet GSR" },
  { value: "−90%", label: "Rung radar", note: "Kalman, sai số +3 cm" },
];

export default function App() {
  const [clips, setClips] = useState<ClipInfo[]>([]);
  const [clip, setClip] = useState<ClipInfo | null>(null);
  const [meta, setMeta] = useState<ClipMeta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [opt, setOpt] = useState<ViewOptions>({ kalman: true, speed: true, trails: true });
  const [config, setConfig] = useState<{ upload_enabled: boolean; max_seconds: number; max_upload_mb: number } | null>(null);

  useEffect(() => {
    fetch("/api/config").then((r) => r.json()).then(setConfig).catch(() => setConfig(null));
  }, []);

  const addUploadedClip = async (clipId: string) => {
    const m: ClipMeta = await fetch(`/api/clips/${clipId}/meta`).then((r) => r.json());
    const c: ClipInfo = { id: clipId, title: "Clip của bạn", fps: m.video.fps, frames: m.video.frames, duration: +(m.video.frames / m.video.fps).toFixed(1) };
    setClips((list) => [...list, c]);
    setClip(c);
  };

  useEffect(() => {
    fetch("/api/clips")
      .then((r) => r.json())
      .then((list: ClipInfo[]) => {
        setClips(list);
        setClip(list[0] ?? null);
      })
      .catch(() => setError("Không tải được danh sách clip"));
  }, []);

  useEffect(() => {
    if (!clip) return;
    setMeta(null);
    fetch(`/api/clips/${clip.id}/meta`)
      .then((r) => r.json())
      .then(setMeta)
      .catch(() => setError("Không tải được thông tin clip"));
  }, [clip]);

  const toggle = (k: keyof ViewOptions) => setOpt((o) => ({ ...o, [k]: !o[k] }));

  return (
    <div className="app">
      <header>
        <div>
          <h1>Football CV Tracking</h1>
          <p className="muted">
            Detection, tracking, phân đội và radar 2D có tốc độ thật. Overlay được vẽ ngay trên trình duyệt từ annotation
            JSON stream qua WebSocket; server không gửi video đã render.
          </p>
        </div>
        <nav>
          <a href={REPO} target="_blank" rel="noreferrer">GitHub</a>
          <a href={`${REPO}/blob/main/docs/blog_chong_rung_overlay.md`} target="_blank" rel="noreferrer">Bài viết chống rung</a>
        </nav>
      </header>

      <section className="metrics">
        {METRICS.map((m) => (
          <div key={m.label} className="metric">
            <div className="value">{m.value}</div>
            <div className="label">{m.label}</div>
            <div className="muted small">{m.note}</div>
          </div>
        ))}
      </section>

      <main>
        <div className="stage">
          {error && <div className="error">{error}</div>}
          {clip && meta ? <ClipPlayer clip={clip} meta={meta} opt={opt} /> : !error && <div className="muted">Đang tải…</div>}
        </div>
        <aside>
          <h2>Clip</h2>
          {clips.map((c) => (
            <button key={c.id} className={`clip ${clip?.id === c.id ? "active" : ""}`} onClick={() => setClip(c)}>
              {c.title} <span className="muted small">{c.duration}s</span>
            </button>
          ))}
          {config?.upload_enabled && (
            <>
              <h2>Clip của bạn</h2>
              <UploadPanel maxSeconds={config.max_seconds} maxMb={config.max_upload_mb} onDone={addUploadedClip} />
            </>
          )}
          <h2>Hiển thị</h2>
          <label className="switch">
            <input type="checkbox" checked={opt.kalman} onChange={() => toggle("kalman")} />
            <span>
              Lọc Kalman trên radar
              <span className="muted small block">Tắt để thấy chấm radar rung khi chưa lọc</span>
            </span>
          </label>
          <label className="switch">
            <input type="checkbox" checked={opt.speed} onChange={() => toggle("speed")} />
            <span>Tốc độ trên nhãn</span>
          </label>
          <label className="switch">
            <input type="checkbox" checked={opt.trails} onChange={() => toggle("trails")} />
            <span>Vệt di chuyển 1 s</span>
          </label>
        </aside>
      </main>

      <footer className="muted small">
        Research demo. Footage từ dataset SoccerNet, chỉ dùng cho mục đích nghiên cứu, phi thương mại.
      </footer>
    </div>
  );
}
