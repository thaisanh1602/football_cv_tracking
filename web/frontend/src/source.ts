import type { ClipInfo, ClipMeta, FrameAnn } from "./types";

/**
 * Nguồn dữ liệu cho frontend, 2 chế độ:
 * - server: FastAPI (REST + WebSocket annotation theo khoảng frame, upload qua job queue)
 * - static: hosting tĩnh không có backend (vd HF Static Space). Đọc clips/index.json, tải trọn clips/<id>.json
 *   (~1 MB cho 30 s) và clips/<id>.mp4. Overlay / radar / bật tắt Kalman y hệt; không có upload.
 * Chọn tự động: gọi được api/config thì là server, không thì static.
 */
export interface UploadConfig {
  upload_enabled: boolean;
  max_seconds: number;
  max_upload_mb: number;
}

export interface Source {
  mode: "server" | "static";
  config: UploadConfig | null;
  listClips(): Promise<ClipInfo[]>;
  getMeta(id: string): Promise<ClipMeta>;
  videoUrl(id: string): string;
  /** Chế độ static: trả toàn bộ frame (đã tải cùng meta). Chế độ server: null -> dùng WebSocket. */
  allFrames(id: string): FrameAnn[] | null;
}

const json = async <T>(url: string): Promise<T> => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
};

function serverSource(config: UploadConfig): Source {
  return {
    mode: "server",
    config,
    listClips: () => json<ClipInfo[]>("api/clips"),
    getMeta: (id) => json<ClipMeta>(`api/clips/${id}/meta`),
    videoUrl: (id) => `api/clips/${id}/video`,
    allFrames: () => null,
  };
}

function staticSource(): Source {
  const docs = new Map<string, ClipMeta & { frames: FrameAnn[] }>();
  return {
    mode: "static",
    config: null,
    listClips: () => json<ClipInfo[]>("clips/index.json"),
    async getMeta(id) {
      if (!docs.has(id)) docs.set(id, await json(`clips/${id}.json`));
      const { frames: _frames, ...meta } = docs.get(id)!;
      return meta;
    },
    videoUrl: (id) => `clips/${id}.mp4`,
    allFrames: (id) => docs.get(id)?.frames ?? null,
  };
}

export async function detectSource(): Promise<Source> {
  try {
    const r = await fetch("api/config");
    const type = r.headers.get("content-type") ?? "";
    if (r.ok && type.includes("application/json")) return serverSource(await r.json());
  } catch {
    // không có backend
  }
  return staticSource();
}
