// src/lib/nlp.ts
import dayjs from "dayjs";

export type ParsedAction = "none" | "clear" | "generate";

export type ParsedTask = {
  id: string;
  title: string;
  subject: string;
  minutes: number;
  difficulty: number;
  due: Date;
};

const SUBJECT_HINTS = [
  "math",
  "history",
  "physics",
  "chemistry",
  "english",
  "biology",
  "cs",
  "economics",
];

const DOW: Record<string, number> = {
  sun: 0, sunday: 0,
  mon: 1, monday: 1,
  tue: 2, tues: 2, tuesday: 2,
  wed: 3, weds: 3, wednesday: 3,
  thu: 4, thurs: 4, thursday: 4,
  fri: 5, friday: 5,
  sat: 6, saturday: 6,
};

function parseMinutes(s: string): number | null {
  const m = s.match(/(\d+)\s*(m|min|mins|minute|minutes)\b/i);
  if (m) return Math.max(10, parseInt(m[1], 10));
  const h = s.match(/(\d+)\s*(h|hr|hrs|hour|hours)\b/i);
  if (h) return Math.max(10, parseInt(h[1], 10) * 60);
  return null;
}

function parseDifficulty(s: string): number | null {
  const m = s.match(/\b(?:diff|difficulty)\s*(\d)\b/i);
  return m ? Math.min(5, Math.max(1, parseInt(m[1], 10))) : null;
}

function nextWeekday(targetDow: number): dayjs.Dayjs {
  const today = dayjs();
  let d = today;
  for (let i = 0; i < 8; i++) {
    if (d.day() === targetDow) return d;
    d = d.add(1, "day");
  }
  return today.add(7, "day");
}

function parseDue(s: string): Date | null {
  const lower = s.toLowerCase();

  // today / tomorrow
  if (/\btoday\b/.test(lower)) return dayjs().endOf("day").toDate();
  if (/\btomorrow\b/.test(lower)) return dayjs().add(1, "day").endOf("day").toDate();

  // next <weekday>
  const nextDow = lower.match(/\bnext\s+(sun|mon|tue|tues|wed|weds|thu|thurs|fri|sat|sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b/);
  if (nextDow) {
    const dow = DOW[nextDow[1]];
    return nextWeekday(dow).endOf("day").toDate();
  }

  // plain <weekday>
  const dowOnly = lower.match(/\b(sun|mon|tue|tues|wed|weds|thu|thurs|fri|sat|sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b/);
  if (dowOnly) {
    const target = DOW[dowOnly[1]];
    const today = dayjs();
    let d = today;
    for (let i = 0; i < 7; i++) {
      if (d.day() === target) {
        // if same-day and already past 23:59, push a week
        if (d.isBefore(today, "day")) d = d.add(7, "day");
        return d.endOf("day").toDate();
      }
      d = d.add(1, "day");
    }
    return today.endOf("week").toDate();
  }

  // “due <Fri 5pm> / <2025/09/12>”
  const duePart = lower.match(/\bdue\s+(.+)$/);
  if (duePart) {
    // crude: try Day.js direct parse (YYYY-MM-DD / YYYY/MM/DD or “Fri 5pm”)
    const parsed = dayjs(duePart[1].trim());
    if (parsed.isValid()) return parsed.toDate();
  }

  return null;
}

function guessSubject(text: string): string {
  const lower = text.toLowerCase();
  const found = SUBJECT_HINTS.find((s) => lower.includes(s));
  return found ?? "general";
}

function stripMeta(text: string): string {
  return text
    .replace(/\bdue\b.+/i, "") // drop “due …”
    .replace(/\b(?:difficulty|diff)\s*\d\b/gi, "")
    .replace(/\b\d+\s*(?:m|min|mins|minute|minutes|h|hr|hrs|hour|hours)\b/gi, "")
    .replace(/\b(today|tomorrow|next\s+\w+|sun|mon|tue|tues|wed|weds|thu|thurs|fri|sat)\b/gi, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function parseChatInput(input: string): { tasks: ParsedTask[]; action: ParsedAction } {
  const raw = input.trim();
  const lower = raw.toLowerCase();

  if (lower === "clear") return { tasks: [], action: "clear" };
  if (lower === "generate") return { tasks: [], action: "generate" };

  const minutes = parseMinutes(raw) ?? 120;
  const difficulty = parseDifficulty(raw) ?? 3;
  const due = parseDue(raw) ?? dayjs().endOf("week").toDate();
  const subject = guessSubject(raw);
  const titlePart = stripMeta(raw) || `${subject} task`;

  const task: ParsedTask = {
    id: crypto.randomUUID(),
    title: titlePart,
    subject,
    minutes,
    difficulty,
    due,
  };

  return { tasks: [task], action: "none" };
}
