// src/lib/planner.ts
// --- at top of src/lib/planner.ts ---
import dayjs, { Dayjs } from "dayjs";
import isSameOrBefore from "dayjs/plugin/isSameOrBefore";
dayjs.extend(isSameOrBefore);
import { nanoid } from "nanoid";
import type { Task, Slot } from "@/types";

const POMO_MIN = 25;
const DAILY_WINDOWS = [
  { startHour: 14, endHour: 16 }, // 14:00–16:00
  { startHour: 19, endHour: 21 }, // 19:00–21:00
];

function blocksNeeded(minutes: number) {
  return Math.max(1, Math.ceil(minutes / POMO_MIN));
}

function* iterateNext7Days(start: dayjs.Dayjs) {
  for (let i = 0; i < 7; i++) {
    yield start.add(i, "day");
  }
}

export function generatePlan(tasks: Task[], dailyMaxBlocks = 5): { todayTop3: Slot[]; weekSlots: Slot[] } {
  const weekSlots: Slot[] = [];
  const todayTop3: Slot[] = [];
  const now = dayjs();

  // sort: earlier due first, then higher difficulty
  const sorted = [...tasks].sort((a, b) => {
    const ad = dayjs(a.due);
    const bd = dayjs(b.due);
    if (ad.isBefore(bd)) return -1;
    if (ad.isAfter(bd)) return 1;
    return b.difficulty - a.difficulty;
  });

  const dayUsage: Record<string, number> = {}; // YYYY-MM-DD -> blocks scheduled

  for (const t of sorted) {
    const needs = blocksNeeded(t.minutes ?? POMO_MIN);
    let scheduled = 0;

    outer: for (const day of iterateNext7Days(now.startOf("day"))) {
      // if this day is after the task due date, stop searching
      if (day.isAfter(dayjs(t.due), "day")) break;

      const key = day.format("YYYY-MM-DD");
      dayUsage[key] ??= 0;
      if (dayUsage[key] >= dailyMaxBlocks) continue;

      for (const win of DAILY_WINDOWS) {
        let cursor = day.hour(win.startHour).minute(0).second(0);
        const windowEnd = day.hour(win.endHour).minute(0).second(0);

        // schedule blocks inside this window
        while (cursor.isBefore(windowEnd) && dayUsage[key] < dailyMaxBlocks) {
          weekSlots.push({
            id: nanoid(),
            title: t.title,
            subject: t.subject,
            minutes: POMO_MIN,
            start: cursor.toISOString(),
            end: cursor.add(POMO_MIN, "minute").toISOString(),
            done: false,
          });

          dayUsage[key]++;
          scheduled++;
          if (scheduled >= needs) break outer;

          cursor = cursor.add(POMO_MIN, "minute");
        }
      }
    }
  }

  // today's top 3 (earliest 3 blocks today)
  const todayKey = dayjs().format("YYYY-MM-DD");
  const todays = weekSlots
    .filter((s) => dayjs(s.start).format("YYYY-MM-DD") === todayKey)
    .sort((a, b) => dayjs(a.start).valueOf() - dayjs(b.start).valueOf())
    .slice(0, 3);
  todayTop3.push(...todays);

  return { todayTop3, weekSlots };
}
