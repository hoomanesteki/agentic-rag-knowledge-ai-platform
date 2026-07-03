"use client";

import InfoPage from "../InfoPage";

// Delivery estimates match the company_knowledge / occasions docs the assistant retrieves, so the
// page and the chatbot never disagree.
const RATES = [
  { name: "Standard", price: "Free over $150, else $8", time: "2 to 8 business days" },
  { name: "Express", price: "$15 flat", time: "1 to 2 business days" },
];
const CITIES = [
  { city: "Vancouver & BC", time: "2 to 3 business days" },
  { city: "Calgary & Prairies", time: "3 to 4 business days" },
  { city: "Toronto & Ontario", time: "4 to 5 business days" },
  { city: "Montreal & Quebec", time: "4 to 5 business days" },
  { city: "US mainland", time: "4 to 8 business days" },
];

export default function ShippingPage() {
  return (
    <InfoPage
      title="Shipping & delivery"
      intro="Ships from our Vancouver studio to every province in Canada and every state in the US. We do not ship elsewhere yet."
      ask="What are my shipping options and how long will delivery take?"
    >
      <div className="info-sec">
        <h3>Rates</h3>
        <table className="sizetable">
          <thead>
            <tr>
              <th>Method</th>
              <th>Cost</th>
              <th>Estimated time</th>
            </tr>
          </thead>
          <tbody>
            {RATES.map((r) => (
              <tr key={r.name}>
                <td>{r.name}</td>
                <td>{r.price}</td>
                <td>{r.time}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="info-sec">
        <h3>Estimated delivery by region</h3>
        <table className="sizetable">
          <thead>
            <tr>
              <th>Destination</th>
              <th>Standard delivery</th>
            </tr>
          </thead>
          <tbody>
            {CITIES.map((c) => (
              <tr key={c.city}>
                <td>{c.city}</td>
                <td>{c.time}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="info-note">
          You get a tracking link by email as soon as your order ships. Weather and carrier delays
          can add a day or two; if something looks stuck, give the assistant your email and it will
          check the status for you.
        </p>
      </div>
    </InfoPage>
  );
}
