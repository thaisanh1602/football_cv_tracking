// Khớp schema version 1 của src/export/annotations.py

export type XY = [number, number];
export type Box = [number, number, number, number];

export interface PlayerAnn {
  id: number;
  t: number; // index màu: 0 đội A, 1 đội B, 2 trọng tài, 3 chưa rõ
  c: number; // class: 0 player, 1 goalkeeper, 2 referee
  b: Box;
  v: number | null; // km/h
  r: XY | null; // vị trí sân thô (m)
  k: XY | null; // vị trí sân đã lọc Kalman (m)
}

export interface FrameAnn {
  f: number;
  h: boolean; // có homography
  p: PlayerAnn[];
  ball: { b: Box; r: XY | null } | null;
}

export interface ClipMeta {
  version: number;
  title?: string;
  video: { fps: number; width: number; height: number; frames: number };
  pitch: { length: number; width: number; vertices: XY[]; edges: [number, number][]; centre_circle_radius: number };
  teams: string[];
}

export interface ClipInfo {
  id: string;
  title: string;
  fps: number;
  frames: number;
  duration: number;
}

export interface ViewOptions {
  kalman: boolean;
  speed: boolean;
  trails: boolean;
}
