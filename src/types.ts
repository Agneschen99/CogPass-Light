// Task that user creates (input)
export type Task = {
  id: string;
  subject: string;           // e.g., Math / Chinese / History
  title: string;             // e.g., Chapter 3 Review
  difficulty: number;        // 1–5 (priority or hardness)
  estimatedMinutes: number;  // total minutes required
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
