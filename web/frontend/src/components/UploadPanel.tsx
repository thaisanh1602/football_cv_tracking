import { useRef, useState } from "react";
import { wsUrl } from "../useAnnotationStream";

interface JobStatus {
  id: string;
  state: string; // queued | started | finished | failed
  stage?: string | null; // transcode | analyze | done
  progress?: number | null;
  position?: number | null;
  clip_id?: string;
  error?: string;
}

const STAGE_LABEL: Record<string, string> = {
  transcode: "Chuẩn hoá video",
  analyze: "Phân tích (detect, track, phân đội, radar)",
  done: "Xong",
};

/** Upload clip -> job chạy nền ở worker; tiến độ nhận qua WebSocket. Xong thì gọi onDone(clip_id). */
export function UploadPanel({ maxSeconds, maxMb, onDone }: { maxSeconds: number; maxMb: number; onDone: (clipId: string) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const busy = status != null && !["finished", "failed"].includes(status.state);

  const upload = async (file: File) => {
    setError(null);
    if (file.size > maxMb * 1e6) {
      setError(`File lớn hơn ${maxMb} MB`);
      return;
    }
    const body = new FormData();
    body.append("file", file);
    const res = await fetch("api/jobs", { method: "POST", body });
    if (!res.ok) {
      setError((await res.json().catch(() => null))?.detail ?? `Lỗi ${res.status}`);
      return;
    }
    const st: JobStatus = await res.json();
    setStatus(st);
    const ws = new WebSocket(wsUrl(`ws/jobs/${st.id}`));
    ws.onmessage = (e) => {
      const s: JobStatus = JSON.parse(e.data);
      setStatus(s);
      if (s.state === "finished" && s.clip_id) onDone(s.clip_id);
      if (s.state === "failed") setError(s.error ?? "Xử lý thất bại");
    };
  };

  return (
    <div className="upload">
      <p className="muted small">
        Thử với clip của bạn: tối đa {maxSeconds} s, {maxMb} MB. Server chạy CPU nên mất vài phút mỗi clip.
      </p>
      <input
        ref={input}
        type="file"
        accept="video/*"
        hidden
        onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
      />
      <button className="primary" disabled={busy} onClick={() => input.current?.click()}>
        {busy ? "Đang xử lý…" : "Tải clip lên"}
      </button>
      {status && busy && (
        <div className="job">
          {status.state === "queued" ? (
            <span className="muted small">Đang xếp hàng{status.position != null ? `, vị trí ${status.position + 1}` : ""}</span>
          ) : (
            <>
              <span className="small">{STAGE_LABEL[status.stage ?? ""] ?? "Đang chạy"}</span>
              <progress value={status.progress ?? 0} max={1} />
              <span className="muted small">{Math.round((status.progress ?? 0) * 100)}%</span>
            </>
          )}
        </div>
      )}
      {error && <div className="error small">{error}</div>}
    </div>
  );
}
