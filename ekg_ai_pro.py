# ekg_ai_pro.py — Paramedik EKG Tedavi Algoritmalari (SB-ASH-Y oncelikli)

MANUEL_ISIMLER = {
    "ANTERIOR_MI": "Anterior STEMI",
    "INFERIOR_MI": "Inferior STEMI",
    "LATERAL_MI": "Lateral STEMI",
    "POSTERIOR_MI": "Posterior STEMI",
    "SAG_V_MI": "Sag Ventrikul MI",
    "YAYGIN_ANTERIOR_MI": "Yaygin Anterior MI",
    "NSTEMI": "NSTEMI / USAP",
    "AF": "Atriyal Fibrilasyon",
    "SVT": "Supraventrikuler Tasikardi",
    "VT": "Ventrikuler Tasikardi",
    "VF": "Ventrikuler Fibrilasyon",
    "BRADIKARDI": "Semptomatik Bradikardi",
    "AV_BLOK": "AV Blok",
    "ASISTOLI": "Asistoli",
    "NORMAL": "Normal Sinuz Ritmi",
    "GENEL": "Genel EKG Degerlendirmesi",
    "SOL_ANA_KORONER": "Sol Ana Koroner Darlik / Yaygin Iskemi",
    "WPW": "WPW / Preexcitasyon",
    "TORSADES": "Torsades de Pointes",
    "PEA": "PEA (Elektriksel Aktivite Olmayan Nabiz)",
}


def _prehospital_only(text: str) -> str:
    """Hastane sonrasi satirlari kes; sadece olay yeri + transport kalsin."""
    import re
    if not text:
        return text
    # --- ile baslayan HASTANEDE/Hastanede/sonrasi bolumu temizle
    text = re.sub(r"(?:\n---)?\n?\s*(?:📚)?\s*HASTANEDE.*", "", text, flags=re.S | re.I)
    text = re.sub(r"(?:\n---)?\n?\s*(?:📚)?\s*HASTANE.*", "", text, flags=re.S | re.I)
    text = re.sub(r"(?:\n---)?\n?\s*(?:📚)?\s*SONRA.*", "", text, flags=re.S | re.I)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


TEDAVI_ALGORITMALARI = {
    "VF": {
        "aciliyeti": "KIRMIZI ALARM - ARREST",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • Sahne guvenligi sagla.\n"
            "   • Bilinc degerlendir; yoksa CPR baslat.\n"
            "   • Defibrilator/analizoru ac, yapistirici yamalari uygula.\n\n"
            "2️⃣ MUDAHALE (SB-ASH-Y-11):\n"
            "   • Hemen baslangic kalp masaji (100-120/dk, derinlik 5-6 cm).\n"
            "   • 30 kompresyon : 2 ventilasyon (sadece ihale varsa).\n"
            "   • En kisa surede sok: 200 J biphasik (veya cihazin onerdigi doz).\n"
            "   • Her sok sonrasi CPR hemen devam et; ritim 2 dk sonra kontrol.\n"
            "   • Adrenalin 1 mg IV/IO her 3-5 dakikada bir.\n"
            "   • Amiodaron 300 mg IV/IO ilk doz; gerekirse 150 mg tekrar.\n"
            "   • H/T (hipovolemi, hipoksi, hidrojen/asidoz, hiper/hypokalemi, hipotermi, \n"
            "     ilac/zehirlenme, tamponat, tansiyon pnomonisi, tromboemboli) dusun.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • ROSC saglanirsa 12 derivasyon EKG, hipotermi protokolune uygun transport.\n"
            "   • Surekli CPR/izlem, gelismis airway destegi."
        ),
    },
    "ASISTOLI": {
        "aciliyeti": "KIRMIZI ALARM - ARREST",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • Sahne guvenligi, bilinc yok.\n"
            "   • Duz hat/asistoli dogrula (kablolar/pil/artefakta dikkat).\n"
            "   • 2 farkli derivasyonda dogrula.\n\n"
            "2️⃣ MUDAHALE (SB-ASH-Y-10):\n"
            "   • Hemen yuksek kaliteli CPR baslat.\n"
            "   • Havayolunu ac, oksijen ver.\n"
            "   • Adrenalin 1 mg IV/IO her 3-5 dk.\n"
            "   • SOK YOK (asistolide sok etkisizdir).\n"
            "   • H/T nedenlerini aktif arastir ve duzelt.\n"
            "   • Lopressorlu EKG izlemi, kapnografi varsa takip.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • ROSC saglanirsa post-arrest bakim merkezine transport."
        ),
    },
    "PEA": {
        "aciliyeti": "KIRMIZI ALARM - ARREST",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • Nabiz yok ama monitorde duzenli elektriksel aktivite var.\n"
            "   • Kisa sireli dogrulama; gecikmeden CPR basla.\n\n"
            "2️⃣ MUDAHALE (SB-ASH-Y-10/11):\n"
            "   • Yuksek kaliteli CPR.\n"
            "   • Adrenalin 1 mg IV/IO her 3-5 dk.\n"
            "   • Tersine cevrilebilir nedenleri hizli tani ve tedavi et.\n"
            "   • Hipovolemi -> sivi; hipoksi -> oksijen/intubasyon;\n"
            "     tamponat -> perikardiosentez; tansiyon pnomonisi -> igne dekompresyonu;\n"
            "     masif pulmoner emboli -> fibrinolitik dusun.\n"
            "   • SOK yok (PEA'de sok etkili degildir).\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • ROSC saglanirsa post-arrest merkezine transport."
        ),
    },
    "VT": {
        "aciliyeti": "KIRMIZI ALARM - ACIL",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • Bilinc, TA, nabiz, solunum degerlendir.\n"
            "   • Genis QRS, duzenli; AV ayrimi ara.\n\n"
            "2️⃣ MUDAHALE (SB-ASH-Y-08):\n"
            "   • Nabiz yoksa arrest algoritmasina gec (defibrilasyon + CPR).\n"
            "   • Nabiz var ama hemodinamik bozukluk varsa: senkronize kardiyoversiyon.\n"
            "   • Hemodinamik stabilse: amiodaron 150-300 mg IV/IO yavas.\n"
            "   • Monomorfik VT icin lidoain alternatif.\n"
            "   • Sedasyon kardiyoversiyon icin hazir bulundur.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Defibrilator/monitor esliginde transport, tekrarlayan VT riski."
        ),
    },
    "TORSADES": {
        "aciliyeti": "KIRMIZI ALARM - ACIL",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • Polimorfik VT, QRS eksen donuyor; QTc uzamis olabilir.\n"
            "   • Nabiz ve hemodinami acil degerlendir.\n\n"
            "2️⃣ MUDAHALE:\n"
            "   • Nabiz yoksa defibrilasyon + CPR.\n"
            "   • Nabiz varsa magnezyum 2 g IV yavas (tekrarlanabilir).\n"
            "   • Bradikardi bagimli ise hiz artir (atropin/temporary pacing).\n"
            "   • QTc uzatan ilaclari durdur.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Surekli monitorizasyon, tekrarlayan aritmi riski."
        ),
    },
    "SVT": {
        "aciliyeti": "SARI ALARM - Hemodinami belirler",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • Hiz 150-250/dk, dar ve duzenli QRS.\n"
            "   • TA, bilinc, solunum, iskemi bulgulari.\n"
            "   • Genis QRS ise VT olasiligini gozden kacirma.\n\n"
            "2️⃣ MUDAHALE (SB-ASH-Y-08):\n"
            "   • Hemodinamik bozukluk varsa: sedelektrik senkronize kardiyoversiyon.\n"
            "   • Stabilse: vazal manevra (valsalva, carotis sinuse masaji kontrendike degilse).\n"
            "   • Adenozin 6 mg hizli IV bolus; 1-2 dk sonra 12 mg; gerekirse tekrar 12 mg.\n"
            "   • Adenozin kontrendike (WPW+AF gibi onemli predenasyonda) dikkat.\n"
            "   • Vagal/adenozin basarisizsa amiodaron veya diltiazem dusun.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • 12 derivasyon EKG ve vital takiple transport."
        ),
    },
    "AF": {
        "aciliyeti": "SARI ALARM - Hemodinami belirler",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • Duzensiz dar QRS, P yok; hiz degisken.\n"
            "   • TA, bilinc, kalp yetmezligi bulgulari.\n\n"
            "2️⃣ MUDAHALE (SB-ASH-Y-08):\n"
            "   • Hemodinamik bozukluk varsa: sedelektrik kardiyoversiyon.\n"
            "   • Stabilse hiz kontrolu: diltiazem veya amiodaron IV.\n"
            "   • Akut koroner sendrom, WPW varsa yonetim degisir.\n"
            "   • Sivi yuklemesi ve elektrolit duzeltmesi.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Transport suresince hiz ve hemodinami izlemi."
        ),
    },
    "WPW": {
        "aciliyeti": "SARI ALARM - Prearrest riski",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • Kisa PR, delta dalgasi, genis QRS.\n"
            "   • AF + hizli anterograd ileti -> VF riski.\n\n"
            "2️⃣ MUDAHALE:\n"
            "   • Hemodinamik bozukluk varsa kardiyoversiyon.\n"
            "   • Stabil tasikardide: prokainamid veya ibutilid tercih;\n"
            "     AV nodu yavaslatan ilaclardan (adenozin, verapamil, diltiazem, digoksin, BB) KACIN.\n"
            "   • Elektrolit duzelt.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Surekli monitorizasyon, hizli transport."
        ),
    },
    "BRADIKARDI": {
        "aciliyeti": "SARI ALARM - Semptom varsa KIRMIZI",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • Hiz < 60/dk ve semptom: hipotansiyon, bilinc degisikligi, sok, iskemi.\n"
            "   • Inferior MI, beta bloker/CCB/digoksin zehirlenmesi akilda bulundur.\n\n"
            "2️⃣ MUDAHALE (SB-ASH-Y-07):\n"
            "   • Atropin 0.5 mg IV her 3-5 dk (maksimum 3 mg).\n"
            "   • Atropine yetersizse transkutan pacing hazirla ve uygula.\n"
            "   • Dopamin 5-20 mcg/kg/dk veya epinefrin 2-10 mcg/dk infuzyon dusun.\n"
            "   • Sivi resusitasyonu gerekirse.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Pacing devam ederken hizli transport."
        ),
    },
    "AV_BLOK": {
        "aciliyeti": "SARI ALARM - Mobitz II/tam blok KIRMIZI",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • PR uzamasi, Mobitz I (Wenckebach), Mobitz II, tam AV blok.\n"
            "   • Hiz, TA, bilinc, semptomlar.\n\n"
            "2️⃣ MUDAHALE (SB-ASH-Y-07):\n"
            "   • Mobitz II veya tam blokta transkutan pacing hazir.\n"
            "   • Atropin tam blokta genelde etkisiz ama denenebilir.\n"
            "   • Hipotansiyon varsa atropin + pacing + sivi/inotrop.\n"
            "   • Inferior MI'de gelisen Mobitz I genellikle gecicidir; yakin takip.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Pacing ile transport, transvenous pacing gerekebilir."
        ),
    },
    "ANTERIOR_MI": {
        "aciliyeti": "KIRMIZI ALARM - ACIL",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • Bilinc, solunum, cilt, TA, nabiz, SpO2.\n"
            "   • 12 derivasyon EKG; V1-V4'te ST elevasyonu.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Aspirin 325 mg cigneme (alerji/dispepsi haric).\n"
            "   • Nitrogliserin sublingual (sistolik TA > 90 mmHg, sag ventrikul MI dislanmali).\n"
            "   • Morfin 2-5 mg IV gerekirse (agri/dispne).\n"
            "   • Oksijen sadece SpO2 < 90 veya solunum skintisi varsa.\n"
            "   • IV access ac, heparin/sitagliptin yuklemesi hastanede; sahada aspirin yeterli.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Hedef kapiya kadinik sure < 90 dk.\n"
            "   • PCI merkezine direkt transport; helikopter gerekebilir."
        ),
    },
    "YAYGIN_ANTERIOR_MI": {
        "aciliyeti": "KIRMIZI ALARM - ACIL (Yuksek Risk)",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • V1-V6 + I, aVL yaygin ST elevasyonu; proksimal LAD.\n"
            "   • Kardiyojenik sok, ani kardiak arrest riski yuksek.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Aspirin cigneme.\n"
            "   • Nitrat dikkatli; TA duserse kes ve sivi ver.\n"
            "   • Hipotansiyon varsa sivi resusitasyonu, inotropik destek.\n"
            "   • Morfin gerekirse dikkatli.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Acil PCI merkezine transport; kardiyojenik sok ekibi haberli."
        ),
    },
    "INFERIOR_MI": {
        "aciliyeti": "KIRMIZI ALARM - ACIL",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • II, III, aVF'de ST elevasyonu.\n"
            "   • Mutlaka V4R cek, sag ventrikul MI ekarte et.\n"
            "   • Nabiz, TA, bradikardi/AV blok ara (vagal zenginlestirme).\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Aspirin cigneme.\n"
            "   • TA < 90 mmHg ise NITROGLISERIN KONTRENDIKE.\n"
            "   • Sag V MI varsa SF 250-500 ml bolus, hipotansiyon devam ederse tekrarla.\n"
            "   • Inferior MI'de vagal yanut; atropin gerekirse.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • PCI merkezine hizli transport."
        ),
    },
    "SAG_V_MI": {
        "aciliyeti": "KIRMIZI ALARM - ACIL",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • Inferior MI + V4R ST elevasyonu >= 1 mm.\n"
            "   • Hipotansiyon, JVD, temiz akciger (Preload bagimli).\n\n"
            "2️⃣ OLAY YERI MUDAHALE:\n"
            "   • NITROGLISERIN KESINLIKLE YASAK.\n"
            "   • Morfin yavas ve dikkatli (preload dusurur).\n"
            "   • SF 250-500 ml bolus; hipotansiyon devam ederse tekrarla.\n"
            "   • Aspirin cigneme.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Hizli transport, sag ventrikul disfonksiyonu riski."
        ),
    },
    "LATERAL_MI": {
        "aciliyeti": "KIRMIZI ALARM - ACIL",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • I, aVL, V5-V6 ST elevasyonu.\n\n"
            "2️⃣ OLAY YERI MUDAHALE:\n"
            "   • Aspirin, nitrat (TA uygunsa), oksijen endikasyonuna gore.\n"
            "   • Morfin gerekirse.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • PCI merkezine direkt transport."
        ),
    },
    "POSTERIOR_MI": {
        "aciliyeti": "KIRMIZI ALARM - ACIL",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • V1-V3 yatay ST depresyonu, belirgin R, dik T.\n"
            "   • V7-V9 cek, elevasyon dogrula.\n\n"
            "2️⃣ OLAY YERI MUDAHALE:\n"
            "   • Aspirin, nitrat dikkatli (TA ve sag V MI dikkate alinarak).\n"
            "   • Morfin gerekirse.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • PCI merkezine transport."
        ),
    },
    "NSTEMI": {
        "aciliyeti": "SARI ALARM - YAKIN TAKIP",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • ST elevasyonu yok; ST depresyonu/T inversiyonu olabilir.\n"
            "   • Agrisi olan, instabil hasta yuksek risk.\n\n"
            "2️⃣ OLAY YERI MUDAHALE:\n"
            "   • Aspirin cigneme (alerji yoksa).\n"
            "   • Nitrat (TA uygunsa).\n"
            "   • Seri EKG 5-10 dk'de bir.\n"
            "   • Instabilse yuksek riskli olarak degerlendir.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Uygun merkeze transport, troponin takibi planlanir."
        ),
    },
    "SOL_ANA_KORONER": {
        "aciliyeti": "KIRMIZI ALARM - ACIL",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • aVR elevasyonu + yaygin ST depresyonu.\n"
            "   • Sol ana koroner veya proksimal LAD kritik darlik.\n"
            "   • Kardiyojenik sok riski yuksek.\n\n"
            "2️⃣ OLAY YERI MUDAHALE:\n"
            "   • Aspirin cigneme.\n"
            "   • Sivi resusitasyonu, inotropik destek dusun.\n"
            "   • Nitrat dikkatli; TA duserse kes.\n"
            "   • Morfin gerekirse.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Acil PCI merkezine transport; kardiyojenik sok ekibi haberli."
        ),
    },
    "NORMAL": {
        "aciliyeti": "YESIL - Acil degil",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • Normal sinuz ritmi, hiz 60-100/dk.\n"
            "   • ST/T patolojik degisiklik yok.\n\n"
            "2️⃣ OLAY YERI MUDAHALE:\n"
            "   • Semptom varsa seri EKG + troponin planla.\n"
            "   • Hastaya gore transport karari."
        ),
    },
    "GENEL": {
        "aciliyeti": "SARI ALARM - Degerlendirme gerekli",
        "algoritma": _prehospital_only(
            "1️⃣ GUVENLIK & DEGERLENDIRME:\n"
            "   • Ritim, hiz, duzenlilik, P-QRS-T sistematik incele.\n"
            "   • EKG kalitesi yetersizse tekrar cek.\n\n"
            "2️⃣ OLAY YERI MUDAHALE:\n"
            "   • Semptom ve hemodinamiye gore mudahale.\n"
            "   • Supheli durumda acil kabul et; seri EKG."
        ),
    },
}


def tedavi_algoritmasi_bul(tani_metni: str):
    """Tani metninden en uygun algoritmayi bulur."""
    if not tani_metni:
        return TEDAVI_ALGORITMALARI["GENEL"]

    t = tani_metni.upper()

    if "VF" in t or "FIBRILASYON" in t:
        return TEDAVI_ALGORITMALARI["VF"]
    if "ASISTOL" in t or "DUZ HAT" in t or "DUZ CIZGI" in t:
        return TEDAVI_ALGORITMALARI["ASISTOLI"]
    if "PEA" in t:
        return TEDAVI_ALGORITMALARI["PEA"]
    if "TORSADES" in t or "POINTES" in t:
        return TEDAVI_ALGORITMALARI["TORSADES"]
    if "SAG VENTRIKUL" in t or "SAG V" in t or "V4R" in t:
        return TEDAVI_ALGORITMALARI["SAG_V_MI"]
    if "YAYGIN ANTERIOR" in t:
        return TEDAVI_ALGORITMALARI["YAYGIN_ANTERIOR_MI"]
    if "POSTERIOR" in t or "POSTERIYOR" in t:
        return TEDAVI_ALGORITMALARI["POSTERIOR_MI"]
    if "ANTERIOR" in t:
        return TEDAVI_ALGORITMALARI["ANTERIOR_MI"]
    if "INFERIOR" in t:
        return TEDAVI_ALGORITMALARI["INFERIOR_MI"]
    if "LATERAL" in t:
        return TEDAVI_ALGORITMALARI["LATERAL_MI"]
    if "NSTEMI" in t or "USAP" in t or "NON-ST" in t:
        return TEDAVI_ALGORITMALARI["NSTEMI"]
    if "SOL ANA" in t or ("AVR" in t and "YAYGIN" in t):
        return TEDAVI_ALGORITMALARI["SOL_ANA_KORONER"]
    if "ATRIYAL FIBRILASYON" in t or " ATRIAL FIBRILASYON" in t or t.endswith("AF") or " AF " in t:
        return TEDAVI_ALGORITMALARI["AF"]
    if "WPW" in t or "PREEXCITASYON" in t or "DELTA" in t:
        return TEDAVI_ALGORITMALARI["WPW"]
    if "SVT" in t or "SUPRAVENTRIKULER" in t:
        return TEDAVI_ALGORITMALARI["SVT"]
    if "VENTRIKULER TASIKARDI" in t or " VT" in t or t.endswith("VT"):
        return TEDAVI_ALGORITMALARI["VT"]
    if "BRADIKARDI" in t or "YAVAS RITIM" in t or "BRADY" in t:
        return TEDAVI_ALGORITMALARI["BRADIKARDI"]
    if "AV BLOK" in t or "BLOK" in t:
        return TEDAVI_ALGORITMALARI["AV_BLOK"]
    if "NORMAL" in t or "SINUZ" in t:
        return TEDAVI_ALGORITMALARI["NORMAL"]

    return TEDAVI_ALGORITMALARI["GENEL"]
