"use client";

import InfoPage from "../InfoPage";

const METHODS = [
  "Visa",
  "Mastercard",
  "American Express",
  "Discover",
  "Apple Pay",
  "Google Pay",
  "PayPal",
  "Shop Pay",
];

export default function PaymentPage() {
  return (
    <InfoPage
      title="Payment"
      intro="All the ways to pay, and how we keep it secure. Prices are in Canadian dollars."
      ask="What payment methods do you accept?"
    >
      <div className="info-sec">
        <h3>Accepted methods</h3>
        <div className="pill-row">
          {METHODS.map((m) => (
            <span key={m} className="pill">
              {m}
            </span>
          ))}
        </div>
      </div>

      <div className="info-sec">
        <h3>Pay over time</h3>
        <p>
          Split any order over $50 into four interest-free installments with Afterpay, chosen at
          checkout. No impact to your credit score to check eligibility.
        </p>
      </div>

      <div className="info-sec">
        <h3>Security</h3>
        <p>
          Payments are processed over an encrypted connection and we never store your full card
          number. Your card is charged when the order ships, not when you place it. This is a
          synthetic demo store, so please do not enter a real card.
        </p>
      </div>
    </InfoPage>
  );
}
