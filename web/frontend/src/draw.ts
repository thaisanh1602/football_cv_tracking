import type { ClipMeta, FrameAnn, ViewOptions, XY } from "./types";

/** Overlay trên video. Canvas có kích thước bằng video gốc nên toạ độ pixel dùng thẳng. */
export function drawOverlay(ctx: CanvasRenderingContext2D, fr: FrameAnn | undefined, meta: ClipMeta, opt: ViewOptions) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (!fr) return;
  const scale = meta.video.height / 1080; // nét vẽ / chữ theo độ phân giải
  for (const p of fr.p) {
    const [x1, , x2, y2] = p.b;
    const color = meta.teams[p.t] ?? "#AAAAAA";
    const cx = (x1 + x2) / 2;
    const rx = Math.max((x2 - x1) / 2, 6);
    // ellipse dưới chân, hở phía trước giống EllipseAnnotator của supervision
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5 * scale;
    ctx.beginPath();
    ctx.ellipse(cx, y2, rx, rx * 0.35, 0, (-45 * Math.PI) / 180, (235 * Math.PI) / 180);
    ctx.stroke();

    const label = opt.speed && p.v != null ? `#${p.id} ${Math.round(p.v)} km/h` : `#${p.id}`;
    ctx.font = `600 ${Math.round(13 * scale)}px system-ui, sans-serif`;
    const w = ctx.measureText(label).width + 10 * scale;
    const h = 18 * scale;
    const ly = y2 + rx * 0.35 + 4 * scale;
    ctx.fillStyle = color;
    roundRect(ctx, cx - w / 2, ly, w, h, 4 * scale);
    ctx.fill();
    ctx.fillStyle = "#000";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, cx, ly + h / 2 + 1);
  }
  if (fr.ball) {
    const [x1, y1, x2] = fr.ball.b;
    const cx = (x1 + x2) / 2;
    const s = 10 * scale;
    ctx.fillStyle = "#FFD700";
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, y1 - 4 * scale);
    ctx.lineTo(cx - s, y1 - 4 * scale - 1.7 * s);
    ctx.lineTo(cx + s, y1 - 4 * scale - 1.7 * s);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
}

const PAD = 4; // mét lề quanh sân trên radar

/** Radar 105x68 m. history: các frame trước (cũ -> mới) để vẽ vệt di chuyển. */
export function drawRadar(ctx: CanvasRenderingContext2D, fr: FrameAnn | undefined, history: FrameAnn[], meta: ClipMeta, opt: ViewOptions) {
  const { length, width, vertices, edges, centre_circle_radius } = meta.pitch;
  const W = ctx.canvas.width;
  const s = W / (length + 2 * PAD);
  const px = ([x, y]: XY): XY => [(x + PAD) * s, (y + PAD) * s];

  ctx.fillStyle = "#1f7a3a";
  ctx.fillRect(0, 0, W, ctx.canvas.height);
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = 1.5;
  for (const [a, b] of edges) {
    const [ax, ay] = px(vertices[a]);
    const [bx, by] = px(vertices[b]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }
  const [cx, cy] = px([length / 2, width / 2]);
  ctx.beginPath();
  ctx.arc(cx, cy, centre_circle_radius * s, 0, 2 * Math.PI);
  ctx.stroke();

  if (!fr || !fr.h) {
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.fillRect(0, 0, W, ctx.canvas.height);
    ctx.fillStyle = "#fff";
    ctx.font = "600 14px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Không đủ vạch sân để dựng radar ở frame này", W / 2, ctx.canvas.height / 2);
    return;
  }
  const pos = (p: { r: XY | null; k: XY | null }) => (opt.kalman ? p.k : p.r);

  if (opt.trails) {
    const tracks = new Map<number, { xy: XY[]; t: number }>();
    for (const h of history) {
      if (!h.h) continue;
      for (const p of h.p) {
        const q = pos(p);
        if (!q) continue;
        const tr = tracks.get(p.id) ?? { xy: [], t: p.t };
        tr.xy.push(q);
        tracks.set(p.id, tr);
      }
    }
    ctx.lineWidth = 2;
    for (const { xy, t } of tracks.values()) {
      if (xy.length < 2) continue;
      ctx.strokeStyle = meta.teams[t] + "99";
      ctx.beginPath();
      xy.forEach((q, i) => {
        const [x, y] = px(q);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
  }

  for (const p of fr.p) {
    const q = pos(p);
    if (!q) continue;
    const [x, y] = px(q);
    ctx.fillStyle = meta.teams[p.t] ?? "#AAAAAA";
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }
  if (fr.ball?.r) {
    const [x, y] = px(fr.ball.r);
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }
}

export function radarHeight(meta: ClipMeta, w: number) {
  return Math.round((w * (meta.pitch.width + 2 * PAD)) / (meta.pitch.length + 2 * PAD));
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
