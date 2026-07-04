"use client";

import InfoPage from "../InfoPage";

const PERKS = [
  { title: "Earn points", body: "1 point per dollar spent, redeemable for discounts at checkout." },
  { title: "Early access", body: "Shop new drops and restocks before everyone else." },
  { title: "Member shipping", body: "Free-shipping thresholds and members-only shipping offers." },
  { title: "Birthday reward", body: "A discount during your birthday month, on us." },
];

export default function MembershipPage() {
  return (
    <InfoPage
      title="Aster Circle membership"
      intro="Free to join, no card required. Earn on every order and get first access to what is new."
      ask="What do I get with Aster Circle membership?"
    >
      <div className="help-grid">
        {PERKS.map((p) => (
          <section key={p.title} className="help-card">
            <h3>{p.title}</h3>
            <p>{p.body}</p>
          </section>
        ))}
      </div>

      <div className="info-sec">
        <h3>First order</h3>
        <p>
          Join the Circle or the newsletter and get 15 percent off your first order, applied
          automatically at checkout. Students, healthcare workers, and first responders get 15
          percent off year round with verification.
        </p>
      </div>
    </InfoPage>
  );
}
