// src/state/store.ts
import { create } from "zustand";
import { nanoid } from "nanoid";
import type { Task as AppTask, Slot as SlotType } from "@/types";

export type Task = AppTask;
export type WeekSlot = SlotType;
export type Plan = { todayTop3?: SlotType[]; weekSlots?: SlotType[] };
export type CheckIn = { at: string; note?: string };

type Store = {
  tasks: Task[];
  pomodoros: number;
  todayTop3: WeekSlot[];
  weekSlots: WeekSlot[];
  checkIns: CheckIn[];

  addTask: (t: Partial<Task> & { title: string }) => void;
  removeTask: (id: string) => void;
  clearTasks: () => void;

  setPlan: (plan: Plan) => void;
  addCheckIn: (c: CheckIn) => void;
  toggleDone: (id?: string | number) => void;
  clearAll: () => void;
};

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
          id: (t as any).id ?? nanoid(),
          title: (t as any).title,
          subject: (t as any).subject,
          difficulty: (t as any).difficulty,
          due: (t as any).due,
          minutes: (t as any).minutes ?? (t as any).estimatedMinutes,
        },
      ],
    })),

  removeTask: (id) =>
    set((s) => ({ tasks: (s.tasks ?? []).filter((task) => task.id !== id) })),

  clearTasks: () => set(() => ({ tasks: [] })),

  setPlan: (plan) =>
    set(() => ({
      todayTop3: plan.todayTop3 ?? [],
      weekSlots: plan.weekSlots ?? [],
    })),

  addCheckIn: (c) =>
    set((s) => ({ checkIns: [...(s.checkIns ?? []), c] })),

  toggleDone: (id?: string | number) =>
    set((s) => {
      if (id === undefined) return {} as any;
      return {
        tasks: (s.tasks ?? []).map((task) =>
          task.id === id ? { ...task, done: !((task as any).done) } : task
        ),
        todayTop3: (s.todayTop3 ?? []).map((t) =>
          (t as any).id === id ? { ...(t as any), done: !((t as any).done) } : t
        ),
        weekSlots: (s.weekSlots ?? []).map((w) =>
          (w as any).id === id ? { ...(w as any), done: !((w as any).done) } : w
        ),
      };
    }),

  clearAll: () =>
    set(() => ({ tasks: [], todayTop3: [], weekSlots: [], plan: { todayTop3: [], weekSlots: [] } })),
}));
