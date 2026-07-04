"use client";

import InfoPage from "../InfoPage";

// Body measurements in inches. Fit notes match what the reviews and occasion guides say, so the
// assistant and the chart agree.
const TOPS = [
  { size: "XS", chest: "31 to 33", waist: "24 to 26" },
  { size: "S", chest: "34 to 36", waist: "27 to 29" },
  { size: "M", chest: "37 to 39", waist: "30 to 32" },
  { size: "L", chest: "40 to 42", waist: "33 to 35" },
  { size: "XL", chest: "43 to 45", waist: "36 to 38" },
];
const BOTTOMS = [
  { size: "XS", waist: "24 to 26", hip: "34 to 36" },
  { size: "S", waist: "27 to 29", hip: "37 to 39" },
  { size: "M", waist: "30 to 32", hip: "40 to 42" },
  { size: "L", waist: "33 to 35", hip: "43 to 45" },
  { size: "XL", waist: "36 to 38", hip: "46 to 48" },
];

export default function SizeGuidePage() {
  return (
    <InfoPage
      title="Size guide"
      intro="Measurements are body measurements in inches, not garment measurements. Between two sizes? Our leggings run compressive and a touch small, so size up for a relaxed feel, and size up in outerwear if you plan to layer."
      ask="I am between two sizes, how should I choose?"
    >
      <div className="info-sec">
        <h3>Tops, sports bras & jackets</h3>
        <table className="sizetable">
          <thead>
            <tr>
              <th>Size</th>
              <th>Chest (in)</th>
              <th>Natural waist (in)</th>
            </tr>
          </thead>
          <tbody>
            {TOPS.map((r) => (
              <tr key={r.size}>
                <td>{r.size}</td>
                <td>{r.chest}</td>
                <td>{r.waist}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="info-sec">
        <h3>Leggings, shorts & joggers</h3>
        <table className="sizetable">
          <thead>
            <tr>
              <th>Size</th>
              <th>Waist (in)</th>
              <th>Hip (in)</th>
            </tr>
          </thead>
          <tbody>
            {BOTTOMS.map((r) => (
              <tr key={r.size}>
                <td>{r.size}</td>
                <td>{r.waist}</td>
                <td>{r.hip}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="info-note">
          Bags and most accessories are one size. Leggings run compressive, so size up if you want a
          relaxed feel. Every product page notes when a piece runs small or large.
        </p>
      </div>
    </InfoPage>
  );
}
