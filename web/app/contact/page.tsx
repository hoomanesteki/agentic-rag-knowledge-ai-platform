"use client";

import InfoPage from "../InfoPage";

// All contact details are synthetic. The phone numbers are deliberately fake (repeated-digit demo
// numbers) so nobody dials a real line.
const CHANNELS = [
  { label: "Phone", value: "+1 (778) 111-1111", note: "Mon to Sun, 8am to 8pm Eastern" },
  { label: "Toll-free", value: "1 (888) 111-1111", note: "Canada & US" },
  { label: "Email", value: "support@aster.demo", note: "Replies within one business day" },
  { label: "Live chat", value: "The assistant, bottom right", note: "Instant, 24/7" },
];

export default function ContactPage() {
  return (
    <InfoPage
      title="Contact & support"
      intro="Real people, seven days a week. The assistant handles most questions instantly and escalates to a human specialist when you ask."
    >
      <div className="help-grid">
        {CHANNELS.map((c) => (
          <section key={c.label} className="help-card">
            <h3>{c.label}</h3>
            <p className="contact-value">{c.value}</p>
            <p className="info-note" style={{ margin: 0 }}>
              {c.note}
            </p>
          </section>
        ))}
      </div>

      <div className="info-sec">
        <h3>Talk to a human</h3>
        <p>
          Type <strong>&quot;talk to a human&quot;</strong> in the assistant and you will be handed
          to a specialist from the Aster team. Share your email and they can pull up your orders and
          pick up right where the assistant left off.
        </p>
        <p className="info-note">
          These are demo contact details for a portfolio project. The phone numbers are not real, so
          please do not call them.
        </p>
      </div>
    </InfoPage>
  );
}
