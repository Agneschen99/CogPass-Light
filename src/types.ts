// Task that user creates (input)
export type Task = {
  id: string;
  subject: string;           // e.g., Math / Chinese / History
  title: string;             // e.g., Chapter 3 Review
  difficulty: number;        // 1–5 (priority or hardness)
  // total minutes required (compatibility: some modules use `estimatedMinutes`, others use `minutes`)
  estimatedMinutes?: number;
  minutes?: number;
  due: string;               // YYYY-MM-DD
};

// One scheduled Pomodoro block
export type Scheduled = {
  id: string;
  taskId: string;            // link to Task.id
  title: string;             // shown in UI
  date: string;              // YYYY-MM-DD
  start: Date;               // actual start time
  minutes: number;           // usually 25
  done: boolean;             // marked complete
};

// Legacy / UI-friendly slot used across planner and UI (ISO strings, optional done)
export type Slot = {
  id?: string | number;
  title?: string;
  subject?: string;
  start: string; // ISO string
  end?: string; // ISO string
  minutes?: number;
  done?: boolean;
};

// Plan for a single day
export type DayPlan = {
  date: string;              // YYYY-MM-DD
  items: Scheduled[];
};

// The whole plan for a week
export type WeekPlan = {
  todayTop3: Scheduled[];
  weekSlots: DayPlan[];
};
