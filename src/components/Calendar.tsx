'use client';

import FullCalendar from '@fullcalendar/react';
import dayGridPlugin from '@fullcalendar/daygrid';
import timeGridPlugin from '@fullcalendar/timegrid';
import interactionPlugin from '@fullcalendar/interaction';

type CalendarProps = {
  events?: any[];
};

export default function Calendar({ events = [] }: CalendarProps) {
  return (
    <div className="rounded-xl border bg-white p-4 shadow-sm">
      <FullCalendar
        plugins={[dayGridPlugin, timeGridPlugin, interactionPlugin]}
        initialView="timeGridWeek"
        selectable
        editable
        headerToolbar={{
          left: 'prev,next today',
          center: 'title',
          right: 'dayGridMonth,timeGridWeek,timeGridDay',
        }}
        events={events}
        eventClassNames={(arg) =>
          arg.event.extendedProps?.kind === 'task'
            ? 'bg-blue-200 text-black'
            : 'bg-green-200 text-black'
        }
      />
    </div>
  );
}
