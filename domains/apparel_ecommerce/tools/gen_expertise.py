#!/usr/bin/env python3
"""Generate a deep 'sales expertise' knowledge file so the assistant answers like an expert stylist.

Covers the angles real shoppers ask from: gifts (by recipient, occasion, budget), specific sports
(pilates, yoga, running, hiking, lifting, spin, tennis, barre, climbing), events (a World Cup watch
party, a marathon, a festival), weather and cities, age and style, trends and influencer looks, and
fit/body guidance. Every doc names real products with prices so the answer is specific and the chat
can show them as cards. Synthetic only.

Run: uv run python domains/apparel_ecommerce/tools/gen_expertise.py
"""
import json
import os

OUT = "domains/apparel_ecommerce/seed/unstructured/expertise.jsonl"

# (id, category tag, text). Kept conversational and specific, naming real catalog products.
DOCS = [
    # ---- Gifts by recipient ----
    ("EX01", "gift", "Gift ideas for a girlfriend or wife: the Aster Flow Legging ($98) with the "
     "matching Aster Studio Sports Bra ($58) is our most-gifted set. For something cozier, the "
     "Aster Cloud Hoodie ($118) is a soft oversized favorite. On a smaller budget, the Aster "
     "Daytrip Belt Bag ($38) or an Aster gift card ($25 to $250) never miss. If you know she runs "
     "or does pilates, add the Aster Momentum Short ($68)."),
    ("EX02", "gift", "Gift ideas for a boyfriend or husband: the Aster Coastal Hoodie ($104) and "
     "the Aster Meridian Jogger ($128) make an easy everyday set. For an active guy, the Aster "
     "Base Shell ($224) rain jacket or the Aster Aurora Jacket ($166) are standout gifts. Smaller "
     "options: the Aster Peak Beanie ($38), Aster Marin Gloves ($26), or an Aster gift card."),
    ("EX03", "gift", "Gift ideas for mom: comfortable and versatile wins. The Aster Cloud Hoodie "
     "($118) and the Aster Everywhere Jogger ($108) are soft, flattering, and easy to wear all "
     "day. If she does yoga or pilates, the Aster Flow Legging ($98) with the Aster Base Merino "
     "Long Sleeve ($88) is a lovely, gentle-on-skin pairing. The Aster Dune Tote ($118) is a "
     "great everyday-bag gift."),
    ("EX04", "gift", "Gift ideas for dad: the Aster Cadence Parka ($232) for cold days or the "
     "Aster Base Shell ($224) for rain are memorable gifts. For everyday, the Aster Coastal "
     "Hoodie ($104) and Aster Boreal Tee ($80). Accessories like the Aster Field Beanie ($48) or "
     "the Aster Cadence Duffel ($112) for the gym also land well."),
    ("EX05", "gift", "Gift ideas for a sister, best friend, or coworker: the Aster Studio Sports "
     "Bra ($58) plus Aster Flow Legging ($98) for someone active, or the cozy Aster Cloud Hoodie "
     "($118). Budget-friendly crowd-pleasers: the Aster Daytrip Belt Bag ($38), Aster Halo Socks "
     "($32), Aster Coastal Cap ($30), or an Aster gift card."),
    ("EX06", "gift", "Gift ideas for a teen or someone in their early 20s: trend-forward and fun. "
     "A matching set of the Aster Flow Legging ($98) and Aster Studio Sports Bra ($58), the Aster "
     "Daytrip Belt Bag ($38) worn crossbody, the Aster Coastal Cap ($30), or the Aster Torrent "
     "Headband ($26). Neutral tones like oatmeal, sand, and black are the safe, on-trend picks."),
    ("EX07", "gift", "Gift ideas for a new mom: soft, forgiving, and easy to move in. The Aster "
     "Cloud Hoodie ($118), the Aster Base Crop Legging ($90) with a high, supportive waist, and "
     "the Aster Solstice Longline Bra ($72) for low-impact comfort. The Aster Base Merino Long "
     "Sleeve ($88) is gentle on sensitive skin."),
    ("EX08", "gift", "Gift ideas for a runner: the Aster Momentum Short ($68) or Aster Rally Bike "
     "Short ($74), the breathable Aster Vent Tech Tee ($78), the Aster Marin Sports Bra ($48) for "
     "support, and the Aster Halo Socks ($32). For race-day layers, add the Aster Vent "
     "Windbreaker ($176)."),
    ("EX09", "gift", "Gift ideas for someone who loves yoga or pilates: the Aster Flow Legging "
     "($98) for buttery four-way stretch, the Aster Studio Sports Bra ($58) or Aster Solstice "
     "Longline Bra ($72) for low-impact support, and the soft Aster Pace Training Top ($56). The "
     "Aster Base Merino Long Sleeve ($88) is perfect for the cool-down."),

    # ---- Gifts by occasion ----
    ("EX10", "occasion", "Birthday gift picks: for her, the Aster Flow Legging ($98) plus Aster "
     "Studio Sports Bra ($58), or the Aster Cloud Hoodie ($118). For him, the Aster Coastal "
     "Hoodie ($104) or the Aster Aurora Jacket ($166). Any budget: the Aster Daytrip Belt Bag "
     "($38) or an Aster gift card. Add free gift-note at checkout."),
    ("EX11", "occasion", "Anniversary gift picks: go a little premium. For her, the Aster Aspen "
     "Parka ($218) or the Aster Cadence Tight ($124) with the Aster Cadence Pullover ($132). For "
     "him, the Aster Base Shell ($224) or Aster Cadence Parka ($232). Pair with a small accessory "
     "like the Aster Marin Gloves ($26) to complete it."),
    ("EX12", "occasion", "Valentine's Day gift picks: a matching Aster Flow Legging ($98) and "
     "Aster Studio Sports Bra ($58) set, the soft Aster Cloud Hoodie ($118), or the Aster "
     "Solstice Longline Bra ($72). Keep it easy with an Aster gift card if you are unsure of size."),
    ("EX13", "occasion", "Mother's Day gift picks: the Aster Cloud Hoodie ($118), the Aster "
     "Everywhere Jogger ($108), or the gentle Aster Base Merino Long Sleeve ($88). The Aster Dune "
     "Tote ($118) is a thoughtful everyday-bag gift."),
    ("EX14", "occasion", "Father's Day gift picks: the Aster Coastal Hoodie ($104), the Aster "
     "Base Shell ($224) rain jacket, or the Aster Cadence Duffel ($112) for the gym. Smaller: the "
     "Aster Field Beanie ($48) or Aster Pace Cap ($32)."),
    ("EX15", "occasion", "Holiday and Christmas gift picks: our most-gifted items are the Aster "
     "Cloud Hoodie ($118), the Aster Flow Legging ($98) plus Aster Studio Sports Bra ($58) set, "
     "the Aster Peak Beanie ($38), and the Aster Daytrip Belt Bag ($38). Gift cards ($25 to $250) "
     "are the safe choice when you do not know their size."),
    ("EX16", "occasion", "Graduation or new-job gift picks: something they will wear every day. "
     "The Aster Everywhere Jogger ($108) with the Aster Cloud Hoodie ($118), or the Aster Commute "
     "Tote ($128) for the office and gym. The Aster Ember Backpack ($54) is a practical, "
     "affordable pick."),

    # ---- Gifts by budget ----
    ("EX17", "budget", "Best gifts under $30: the Aster Glacier Beanie ($18), Aster Marin Gloves "
     "($26), Aster Torrent Headband ($26), Aster Coastal Cap ($30), or Aster Boreal Socks ($20). "
     "An Aster gift card starts at $25."),
    ("EX18", "budget", "Best gifts under $50: the Aster Daytrip Belt Bag ($38), Aster Peak Beanie "
     "($38), Aster Storm Tank ($48), Aster Marin Sports Bra ($48), Aster Field Beanie ($48), or "
     "Aster Halo Socks ($32)."),
    ("EX19", "budget", "Best gifts under $75: the Aster Studio Sports Bra ($58), Aster Vent Tech "
     "Tee ($78) is just over, the Aster Momentum Short ($68), Aster Rally Bike Short ($74), Aster "
     "Ember Backpack ($54), or the Aster Zephyr Belt Bag ($62)."),
    ("EX20", "budget", "Best gifts around $100: the Aster Flow Legging ($98), Aster Everywhere "
     "Jogger ($108), Aster Coastal Hoodie ($104), Aster Cloud Hoodie ($118), or the Aster Kodiak "
     "Sling ($98)."),
    ("EX21", "budget", "Premium and luxury gifts: the Aster Trailhead Jacket ($262), Aster Drift "
     "Puffer ($254), Aster Lumen Parka ($258), Aster Dune Jacket ($266), or the Aster Solstice "
     "Duffel ($150). These are our most technical, longest-lasting pieces."),

    # ---- Sport specific ----
    ("EX22", "sport", "What to wear for pilates or barre: you want a smooth, high-rise legging "
     "that stays put through every roll-up. The Aster Flow Legging ($98) and the Aster Cadence "
     "Tight ($124) are ideal, paired with the low-impact Aster Solstice Longline Bra ($72) or "
     "Aster Studio Sports Bra ($58) and a soft Aster Pace Training Top ($56). Grippy Aster Halo "
     "Socks ($32) help on the reformer."),
    ("EX23", "sport", "What to wear for yoga or hot yoga: breathable and sweat-wicking. The Aster "
     "Flow Legging ($98) or the cropped Aster Base Crop Legging ($90), the Aster Studio Sports "
     "Bra ($58), and the airy Aster Vent Tech Tee ($78) or Aster Storm Tank ($48). For hot yoga, "
     "stay light and skip heavy layers."),
    ("EX24", "sport", "What to wear for running: the breathable Aster Vent Tech Tee ($78) or Aster "
     "Flow Tank ($64), the Aster Momentum Short ($68) or Aster Rally Bike Short ($74), the "
     "supportive Aster Marin Sports Bra ($48) or Aster Summit Sports Bra ($72) for higher impact, "
     "and cushioned Aster Halo Socks ($32). For chilly or wet runs, layer the Aster Vent "
     "Windbreaker ($176)."),
    ("EX25", "sport", "What to wear for the gym and weight training: the compressive Aster Cadence "
     "Tight ($124) or Aster Flow Legging ($98), a high-support Aster Summit Sports Bra ($72), and "
     "the Aster Pace Training Top ($56). Men: the Aster Vent Track Pant ($120) or Aster Cloud "
     "Short ($74) with the Aster Trailhead Training Top ($76). Carry it in the Aster Cadence "
     "Duffel ($112)."),
    ("EX26", "sport", "What to wear for HIIT or crossfit: high-impact support and pieces that move "
     "with you. The Aster Summit Sports Bra ($72) or Aster Rally Bra ($72), the Aster Cadence "
     "Tight ($124) or Aster Rally Bike Short ($74), and the quick-drying Aster Vent Tech Tee "
     "($78)."),
    ("EX27", "sport", "What to wear for spin or cycling: padded-friendly bike shorts like the "
     "Aster Rally Bike Short ($74) or Aster Coastal Bike Short ($70), a supportive Aster Marin "
     "Sports Bra ($48), and the moisture-wicking Aster Pace Training Top ($56). Bring a towel and "
     "the Aster Torrent Headband ($26)."),
    ("EX28", "sport", "What to wear for hiking: the durable Aster Trailhead Pant ($126) or Aster "
     "Everywhere Jogger ($108), the Aster Base Merino Long Sleeve ($88) that regulates "
     "temperature, and a packable Aster Storm Shell Jacket ($178) for weather. Add the Aster "
     "Ember Backpack ($54) and Aster Boreal Socks ($20)."),
    ("EX29", "sport", "What to wear for tennis or pickleball: the Aster Momentum Short ($68) or a "
     "skort-style bike short like the Aster Halo Short ($70), the Aster Vent Tech Tee ($78), and "
     "the Aster Studio Sports Bra ($58). The Aster Pace Cap ($32) keeps the sun off."),
    ("EX30", "sport", "What to wear for walking or everyday movement: comfort-first. The Aster "
     "Everywhere Jogger ($108) or Aster Flow Legging ($98), the soft Aster Cloud Hoodie ($118), "
     "and cushioned Aster Halo Socks ($32). The Aster Daytrip Belt Bag ($38) keeps your phone and "
     "keys hands-free."),
    ("EX31", "sport", "What to wear for climbing or bouldering: stretchy, unrestrictive pieces. "
     "The Aster Base Crop Legging ($90) or Aster Vertex Short ($58) for men, the Aster Marin "
     "Sports Bra ($48), and the Aster Pace Training Top ($56)."),

    # ---- Weather, season, city ----
    ("EX32", "weather", "What to wear in hot, humid summer weather (like a Toronto or New York "
     "July): breathable, light pieces. The Aster Vent Tech Tee ($78), Aster Momentum Short ($68), "
     "and Aster Studio Sports Bra ($58). Quick-drying fabrics keep you cool; skip heavy layers."),
    ("EX33", "weather", "What to wear for a rainy commute: the fully waterproof Aster Storm Shell "
     "Jacket ($178) or Aster Nimbus Shell ($174), over the Aster Base Merino Long Sleeve ($88). "
     "Keep your bag dry in the water-resistant Aster Commute Tote ($128)."),
    ("EX34", "weather", "What to wear for cold winter (like Calgary or Montreal): layer a base, a "
     "mid, and a shell. Start with the Aster Base Merino Long Sleeve ($88), add the Aster Cloud "
     "Hoodie ($118) or Aster Trailhead Puffer ($228), and top with the Aster Aspen Parka ($218). "
     "Finish with the Aster Peak Beanie ($38) and Aster Marin Gloves ($26)."),
    ("EX35", "weather", "What to wear for transitional spring and fall weather: the Aster Vent "
     "Windbreaker ($176) or Aster Cloud Jacket ($180), the Aster Base Merino Long Sleeve ($88), "
     "and the Aster Everywhere Jogger ($108). Easy to add or shed a layer as it warms up."),
    ("EX36", "weather", "Dressing for Vancouver: it is mild and wet, so a great rain shell is "
     "everything. The Aster Storm Shell Jacket ($178) over the Aster Cloud Hoodie ($118), with "
     "the Aster Everywhere Jogger ($108). For men, the Aster Base Shell ($224)."),
    ("EX37", "weather", "Dressing for LA: mild and dry with cool evenings. Light layers win. The "
     "Aster Vent Tech Tee ($78) or Aster Flow Tank ($64) by day, with the Aster Cloud Hoodie "
     "($118) or Aster Vent Windbreaker ($176) for the evening. A heavy winter coat is overkill."),

    # ---- Events ----
    ("EX38", "event", "What to wear to a World Cup or FIFA watch party: comfortable athleisure you "
     "can cheer in. A matching Aster Everywhere Jogger ($108) and Aster Cloud Hoodie ($118), or "
     "the Aster Flow Legging ($98) with an oversized Aster Base Hoodie ($132). Add the Aster "
     "Coastal Cap ($30)."),
    ("EX39", "event", "What to wear on marathon or race day: tested layers you have run in before. "
     "The Aster Vent Tech Tee ($78), Aster Momentum Short ($68), Aster Summit Sports Bra ($72), "
     "and Aster Halo Socks ($32), with the packable Aster Vent Windbreaker ($176) for the start "
     "line. Nothing new on race day."),
    ("EX40", "event", "What to wear to a music festival: hands-free and weatherproof. The Aster "
     "Daytrip Belt Bag ($38) or Aster Kodiak Sling ($98), the Aster Rally Bike Short ($74) or "
     "Aster Momentum Short ($68), a comfy Aster Storm Tank ($48), and the Aster Coastal Cap ($30) "
     "for sun."),
    ("EX41", "event", "What to wear for a travel or airport day: soft, layerable, and comfy for "
     "hours. The Aster Everywhere Jogger ($108), the Aster Cloud Hoodie ($118), and the Aster "
     "Base Merino Long Sleeve ($88) that resists wrinkles and odor. Carry the Aster Ember "
     "Backpack ($54) or Aster Commute Tote ($128)."),
    ("EX42", "event", "What to wear for working from home or errands: the relaxed Aster Everywhere "
     "Jogger ($108) or Aster Base Crop Legging ($90), the Aster Cloud Hoodie ($118), and the "
     "Aster Daytrip Belt Bag ($38) for quick trips out."),
    ("EX43", "event", "What to wear for brunch or casual athleisure: put-together but comfy. The "
     "Aster Cadence Tight ($124) with the Aster Base Merino Long Sleeve ($88), or the Aster "
     "Everywhere Jogger ($108) with a fitted Aster Vent Tech Tee ($78). Neutral tones read "
     "polished."),

    # ---- Trends, celebrity, influencer ----
    ("EX44", "trend", "What is trending right now: matching sets (a legging with a coordinating "
     "sports bra and light layer), neutral earth tones like oatmeal, sand, and olive, and "
     "lightweight packable rain shells. The Aster Flow Legging ($98) with the Aster Studio Sports "
     "Bra ($58), and the Aster Storm Shell Jacket ($178) are our on-trend hero pieces."),
    ("EX45", "trend", "The 'clean girl' and minimalist aesthetic: simple, tonal, and elevated. "
     "Stick to black, charcoal, oatmeal, and sand. The Aster Cadence Tight ($124), Aster Base "
     "Merino Long Sleeve ($88), and Aster Everywhere Jogger ($108) nail the look. Skip loud "
     "logos."),
    ("EX46", "trend", "Influencer and celebrity athleisure looks: the off-duty model uniform is a "
     "high-rise legging, a longline bra peeking under an oversized hoodie, and a crossbody bag. "
     "Recreate it with the Aster Flow Legging ($98), Aster Solstice Longline Bra ($72), Aster "
     "Cloud Hoodie ($118), and Aster Daytrip Belt Bag ($38)."),
    ("EX47", "trend", "Most popular colors this season: black, storm blue, oatmeal, and sand. For "
     "a polished, put-together look, keep your set tonal in one of these and let the texture do "
     "the talking, like the Aster Cadence Pullover ($132) over the Aster Cadence Tight ($124)."),

    # ---- Fit, body, materials ----
    ("EX48", "fit", "If you are petite or shorter: our leggings come in a 25-inch and 28-inch "
     "inseam feel depending on style; the cropped Aster Base Crop Legging ($90) and Aster Kodiak "
     "Crop Legging ($94) hit at a flattering length. Size down in leggings for compression."),
    ("EX49", "fit", "If you are tall or want a longer inseam: the full-length Aster Flow Legging "
     "($98), Aster Dune Legging ($122), and Aster Everywhere Jogger ($108) have the most length. "
     "We do not carry a dedicated tall line yet, but these run long."),
    ("EX50", "fit", "If you are curvy or between sizes: our high-rise waistbands like the Aster "
     "Cadence Tight ($124) and Aster Base Crop Legging ($90) are supportive and stay put. Size up "
     "in outerwear for layering; size down in leggings for a compressive feel. Sizes run XS to "
     "XL."),
    ("EX51", "fit", "For sensitive skin: choose soft, tag-light natural fibers. The Aster Base "
     "Merino Long Sleeve ($88) is fine, non-itch merino, and our cotton-blend tees like the Aster "
     "Torrent Tee ($58) are gentle. Our performance fabrics are smooth and seam-flat. Skip fabric "
     "softener, which can irritate skin."),
    ("EX52", "fit", "Highest-support sports bras for larger busts or high-impact training: the "
     "Aster Summit Sports Bra ($72), Aster Rally Bra ($72), and Aster Ember Bra ($66). For "
     "low-impact yoga or lounging, the Aster Solstice Longline Bra ($72) or Aster Zephyr Longline "
     "Bra ($50)."),
    ("EX53", "materials", "Fabric and feel guide: the Aster Cloud Hoodie ($118) has a soft "
     "brushed-fleece interior, the Aster Flow Legging ($98) is smooth compressive four-way "
     "stretch, the Aster Base Merino Long Sleeve ($88) is fine non-itch merino, and the Aster "
     "Storm Shell Jacket ($178) is a crisp waterproof shell. If you like buttery-soft, go for the "
     "Cloud Hoodie or the merino base layer."),

    # ---- Reworded / synonym coverage so a rephrase still lands ----
    ("EX54", "weather", "Warm leggings for cold winter weather: our leggings are mid-weight, not "
     "fleece-lined, so for real warmth layer them. The thick, high-rise Aster Cadence Tight ($124) "
     "or Aster Dune Legging ($122) hold heat best, and you can pull the Aster Everywhere Jogger "
     "($108) over them or add the Aster Base Merino Long Sleeve ($88) on top for a base layer. For "
     "the coldest days a lined track pant like the Aster Boreal Track Pant ($138) is warmer than "
     "any legging."),
    ("EX55", "sport", "What to wear for rock climbing, indoor climbing, gym climbing, or "
     "bouldering: stretchy, unrestrictive pieces that move with big reaches. Women: the Aster Base "
     "Crop Legging ($90) with the Aster Marin Sports Bra ($48) and the soft Aster Pace Training "
     "Top ($56). Men: the Aster Vertex Short ($58) or Aster Cloud Short ($74) with the Aster Flow "
     "Tank ($64)."),
    ("EX56", "fit", "Styles for a curvy, fuller, or plus figure (we run XS to XL): high-rise, "
     "supportive waistbands that stay put, like the Aster Cadence Tight ($124), Aster Base Crop "
     "Legging ($90), and Aster Dune Legging ($122). For tops, the relaxed Aster Cloud Hoodie "
     "($118) and Aster Base Hoodie ($132). Size up in outerwear for layering; our high-support "
     "Aster Summit Sports Bra ($72) and Aster Ember Bra ($66) give the most lift."),
    ("EX57", "fit", "Style for a woman in her 40s, over 40, or midlife: elevated, tonal, and "
     "comfortable. The Aster Cadence Tight ($124) with the Aster Base Merino Long Sleeve ($88), or "
     "the Aster Everywhere Jogger ($108) with the Aster Cloud Hoodie ($118). Neutral colors like "
     "black, charcoal, and oatmeal read timeless and put-together."),
    ("EX58", "fit", "Workout clothes to wear while pregnant or expecting (maternity, pregnancy, "
     "postpartum): we do not carry a dedicated maternity line, but several pieces are bump-friendly "
     "and forgiving for working out through pregnancy. The high-rise Aster Base Crop Legging ($90) "
     "and Aster Cadence Tight ($124) have a wide, comfortable waistband that sits under a bump, and "
     "the relaxed Aster Cloud Hoodie ($118) and soft Aster Base Merino Long Sleeve ($88) work while "
     "pregnant and after. A gift card is a safe choice if sizing is uncertain."),
    ("EX59", "catalog", "Categories we do not carry: we do not sell shoes, swimwear, socks aside "
     "from athletic socks, denim, or team jerseys. We focus on athletic apparel, outerwear, bags, "
     "and accessories. If you need one of those, we can still help with what to pair it with, like "
     "the Aster Vent Tech Tee ($78) or Aster Momentum Short ($68)."),

    # ---- More varied gift anchors, so back-to-back gift asks are not identical ----
    ("EX60", "gift", "Gift ideas for a teenage girl or Gen-Z shopper: the crossbody Aster Daytrip "
     "Belt Bag ($38), the Aster Coastal Cap ($30), the Aster Torrent Headband ($26), or a fun "
     "cropped set with the Aster Base Crop Legging ($90). Trend-forward neutrals and a gift card "
     "are safe if you are unsure of size."),
    ("EX61", "gift", "Gift ideas for a best friend: the cozy Aster Base Hoodie ($132) or Aster "
     "Aurora Pullover ($130), the Aster Dune Tote ($118) for everyday, or a matching accessory "
     "duo of the Aster Peak Beanie ($38) and Aster Marin Gloves ($26)."),
    ("EX62", "gift", "Gift ideas for a wife who has everything: go premium and personal, like the "
     "Aster Aspen Parka ($218), the Aster Cadence Pullover ($132) with the Aster Cadence Tight "
     "($124), or the Aster Solstice Duffel ($150) for weekend trips. Pair with the Aster Storm "
     "Scarf ($34) to finish it."),
    ("EX63", "gift", "Gift ideas for a coworker or a Secret Santa under budget: the Aster Daytrip "
     "Belt Bag ($38), Aster Coastal Cap ($30), Aster Halo Socks ($32), Aster Torrent Headband "
     "($26), or an Aster gift card. Safe, useful, and no sizing worries."),
    ("EX64", "gift", "Gift ideas for a sister: the Aster Cloud Hoodie ($118) in a neutral tone, "
     "the Aster Everywhere Jogger ($108), or the Aster Kodiak Sling ($98) crossbody bag. For "
     "someone sporty, the Aster Marin Sports Bra ($48) with the Aster Momentum Short ($68)."),
]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for cid, cat, text in DOCS:
            f.write(json.dumps({"id": cid, "lang": "en", "category": cat, "text": text}) + "\n")
    print(f"wrote {len(DOCS)} expertise docs -> {OUT}")


if __name__ == "__main__":
    main()
