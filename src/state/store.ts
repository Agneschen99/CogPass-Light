// src/state/store.ts
import { create } from "zustand";
import { nanoid } from "nanoid";

// ====== Types ======
export type Task = {
  id: string;
  title: string;
  subject?: string;
  difficulty?: number;
  due?: string;         // ISO string
  minutes?: number;
};

export type WeekSlot = {
  id?: string | number;
  title?: string;
  subject?: string;
  start: string;        // ISO string
  end: string;          // ISO string
};

export type Plan = {
  todayTop3?: Task[];
  weekSlots?: WeekSlot[];
};

export type CheckIn = {
  at: string;           // ISO string
  note?: string;
};

type Store = {
  // ----- state -----
  tasks: Task[];
  pomodoros: number;
  todayTop3: Task[];
  weekSlots: WeekSlot[];
  checkIns: CheckIn[];

  // ----- actions -----
  addTask: (t: Omit<Task, "id">) => void;
  removeTask: (id: string) => void;
  clearTasks: () => void;

  setPlan: (plan: Plan) => void;
  addCheckIn: (c: CheckIn) => void;
};

// ====== store ======
export const useStore = create<Store>((set) => ({
  // ---- SAFE DEFAULTS ----
  tasks: [],
  pomodoros: 0,
  todayTop3: [],
  weekSlots: [],
  checkIns: [],

  // ---- TASKS ----
  addTask: (t) =>
    set((s) => ({
      tasks: [
        ...(s.tasks ?? []),
        {
          id: nanoid(),
          title: t.title,
          subject: t.subject,
          difficulty: t.difficulty,
          due: t.due,
          minutes: t.minutes,
        },
      ],
    })),

  removeTask: (id) =>
    set((s) => ({ tasks: (s.tasks ?? []).filter((task) => task.id !== id) })),

  clearTasks: () => set(() => ({ tasks: [] })),

  // ---- PLAN ----
  setPlan: (plan) =>
    set(() => ({
      todayTop3: plan.todayTop3 ?? [],
      weekSlots: plan.weekSlots ?? [],
    })),

  // ---- CHECKINS ----
  addCheckIn: (c) =>
    set((s) => ({
      checkIns: [...(s.checkIns ?? []), c],
    })),
}));
