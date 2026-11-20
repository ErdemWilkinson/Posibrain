# Arama dronu simülasyonu — tasarım (2026-11-20)

Durum: ilk tasarım notu (2026-11-20). Uygulama bu tasarımdan bazı noktalarda sapıyor (örn. 'eve dön' eylemi yok, çoklu dron çalışmıyor). Güncel durum ve sonuçlar için README.md'ye bakın.

## Niyet ve neden

Erdem, ucuz bir arama-kurtarma dronunun "beyninin" bir kısmını sıfırdan öğrenen küçük bir model olarak yapmak istiyor ve bunu önce simülasyonda denemek istiyor. Simülasyon gerçek dronu riske atmadan, bir gecede ölçülebilir sonuç üretmeli. Öğrenilen şey **arama stratejisi**: nereye uçmalı, nerede durup taramalı, bataklıktan ve kuştan nasıl kaçmalı, pil bitmeden insanı nasıl bulmalı. Sonunda aynı politika nano (ESP32-S3 sınıfı) için küçültülüp int8'e nicemlenecek.

## Hedefler

- 10 km × 10 km'lik, her bölümde yeniden üretilen gerçekçi doğa: orman (gölgelik yoğunluğu), bataklık, kaya, rüzgar, yırtıcı kuşlar.
- Her bölümde insanın yeri değişir. Politika harita ezberleyemez.
- Sıfır ağırlıktan eğitim. Baz çizgilerle (rastgele uçuş, klasik tarama desenleri) ölçülü karşılaştırma.
- Gözlemler gerçek nano donanımının üretebileceği şeylerle sınırlı. Politika küçük kalır (hedef: int8'de yüzlerce KB'ın altı, tam sınır ölçümle belirlenecek).
- Her sonuç güven aralığıyla raporlanır (Erdem'in önceki ölçüm alışkanlığı).

## Hedef olmayanlar

- Uçuş aerodinamiği ve denge (PID zaten bunu yapıyor).
- Çamurdan kurtulma. Bataklık burada sadece "yavaşlatan / sıkışma riski olan bölge", olasılıkları uydurma. Gerçek çamur fiziği güvenilir değil, gerçeğe aktarılmaz.
- Radar ve ses sensörleri (v1'de yok).
- Unity/Unreal dünyası (sonraki aşama, aşağıya bak).

## Kötü sonuç sayılacak durumlar (anti-hedefler)

- Politika yalnızca eğitimdeki tohumları (seed) ezberlemiş.
- Simülasyon sonuçlarının gerçek dünyaya aktarıldığı iddiası. Bu, ölçülmüş sensör modeli olmadan yapılmamalı.
- Nano'ya sığmayan ya da nicemlemeye uymayan bir mimari.

## Kısıtlar

- Makine: RTX 4050 Laptop (6 GB), 32 GB RAM, 16 mantıksal çekirdek, Windows. Varsayılan Python 3.11'de yalnızca numpy var. PyTorch kurulu değil (başka bir ortamda olabilir, aranacak). Geceyi etkileyen önceki hatalar için hafızadaki "overnight-training-pitfalls" notu okunmalı (bellek sıkışması, MSYS ortam hatası, export edilen değişkenler).
- Gece bütçesi: tek bir gece. Bu yüzden kapsam dar tutuldu.

## Yaklaşımlar

| | Nasıl | Artı | Eksi |
|---|---|---|---|
| A. Unity/Unreal tam görsel | 10 km 3B dünya, ML-Agents | Güzel, gerçek kamera görüntüsü | Yavaş, bir gecede kurulmaz, eğitim günler sürer |
| B. Hızlı görsel olmayan simülasyon | Python, prosedürel arazi, soyut sensörler | Saniyede binlerce adım, bu gece çalışır, nicemlemeye uygun | Görüntü gerçek değil |
| C. B ile öğren, A'yı sonra gösteri için kullan | B'de eğit, sonra görsel ortamda oynat | İkisinin iyisi | İki iş |

**Erdem C'yi seçti.** Bu gece B yapılır, A ileriye bırakılır.

## Tasarım (Erdem'in kararları ve benim çağrılarım ayrı işaretli)

**Dünya.** 10 m'lik hücrelerle 1000×1000 ızgara. Katmanlar: yükseklik, gölgelik yoğunluğu, bataklık, kaya. Gürültü tabanlı prosedürel üretim. Her bölümde yeni tohum. *Benim çağrım:* eğitim kademeli: önce 1 km, sonra 2, 5 ve 10 km. Sebep: ödülün seyrek olduğu bir problemde 10 km'den başlamak öğrenmeyi çok zorlaştırır.

**İnsan.** Her bölümde rastgele konum. *Benim çağrım:* %70 gölgelik altında, %30 açıkta (daha zor ve gerçekçi).

**Dron.** Nokta kütleli soyut model: konum, yön, hız sınırı, pil (hareket ve havada kalma maliyeti), rüzgar etkisi. Eylemler ayrık: 8 yönde adım, "dur ve tara" (zaman ve pil harcar), "eve dön". *Benim çağrım:* bu, mevcut "dur, tara, ilerle" arama yöntemine uyuyor.

**Sensör (v1).** Aşağı bakan tarama: tespit olasılığı irtifaya, gölgeliğe, tarama süresine ve gündüz/gece durumuna bağlı. Termal benzeri etki: gölgelikte biraz daha iyi, ama yırtıcı kuşlar yanlış alarm üretir. Bunlar *soyut olasılık modelleri*, ölçülmüş değil. Raporlarda böyle etiketlenir.

**Tehlikeler.** Bataklık: yavaşlatır ve sıkışma riski taşır (pil kaybı). Yırtıcı kuş: dronu rahatsız eder, kaçınma gerektirir, termal yanlış alarmı çıkarır. Rüzgar: sürükler.

**Gözlem.** Küçük yerel harita yaması (keşfedilmiş maske, bildiği gölgelik/bataklık bilgisi), pil, eve göre konum, zaman. Hepsi nano'nun gerçekten bilebileceği şeyler (GPS, kendi tuttuğu küçük harita, kamera/termal). *Benim çağrım:* tam harita bilgisi gözleme verilmez.

**Ödül.** İnsanı bulunca büyük artı (hızlı bulana daha çok), her adımda küçük pil cezası, tehlikelere ceza, pil bitmeden eve dönmeye bonus. Ayrıntılar kurucuya bırakıldı.

**Baz çizgiler.** Rastgele yürüyüş, tarama deseni (boustrophedon), spiral, gölgeliğe ağırlıklı tarama (basit kural).

**Ölçüm.** Eğitimde hiç görülmemiş en az 1000 tohumda: bulma oranı (%95 güven aralığıyla), ortalama bulma süresi, kalan pil, tehlike olayı sayısı. Öğrenilen politika baz çizgiyi ancak güven aralıkları ayrışırsa "geçti" sayılır.

**Görselleştirme.** Üstten görünüşlü basit bir oynatıcı (bölümü tekrar oynatır). Unity sonraya.

**Nicemleme.** Küçük MLP/CNN politika → PyTorch → ONNX → int8. KB ve ms ölçümü nano'da yapılır, bu gece değil.

## Bu gece neyi hedefliyoruz

Ortam + baz çizgiler + ölçüm betiği + oynatıcı + ilk eğitim koşusu. Öğrenilen politikanın baz çizgiyi geçmesi **garanti değil**, geçmezse bunu açıkça raporlarız.

## Gözden geçirme sonrası eklenenler (benim çağrılarım, Erdem düzeltebilir)

- **Ölçek sorunu.** Scout benzeri bir dron (hesapla ~20 dk uçuş, ~10 m/s, ~60 m tarama şeridi; hepsi hesap, ölçülmemiş) tek pille yaklaşık 12 km yol alır ve 100 km²'lik haritanın yaklaşık %0,7'sini tarar. İpucusuz 10 km'de herkes ~%0 çıkar. Bu yüzden 10 km dünya olarak kalır, ama bölümün arama alanı daha küçük olmalı (aşağıdaki "açık karar").
- **Sayılar ayarlanabilir.** Hız, pil, şerit genişliği yapılandırma dosyasında. Ayar ilkesi: 1 km kademesinde en iyi baz çizgi yaklaşık %75-80 bulsun. Varsayılan sayılarda mükemmel kapsama tavanı ~%72 olduğundan pil, hız ve şerit buna göre büyütülür. Böylece %60-70 geçiş kapısı ulaşılabilir ama baz çizgiyi geçmeyi gerektirir.
- **İrtifa.** Üç ayrık seviye (alçak/orta/yüksek) eylem olarak eklenir: alçak = daha iyi tespit, dar şerit; yüksek = geniş şerit, zayıf tespit.
- **Yanlış alarm.** Dron "bulundu" bildirir. Kuştan gelen sahte tespit bildirilirse büyük ceza yenir ve bölüm devam eder (pil kaybı). Doğrulamak için tekrar tarayabilir.
- **Bellek.** Dron düşük çözünürlüklü (örn. 100×100) küresel bir "keşfedildi" maskesi tutar (birkaç KB) ve politika tekrarlı değil, düz bir ağdır. Bu hem nano'ya sığmayı hem de nicemlemeyi kolaylaştırır.
- **"Geçti" ölçütü.** Birincil: aynı pil bütçesiyle bulma oranı (güven aralığıyla). İkincil: süre. Tehlike olayları bir kısıt olarak raporlanır. Kapı her kademede ayrı, baz çizgiler aynı bütçeyle.

## Erdem'in kararları (sonradan)

- **İpucusuz.** Dron insanın yerini bilmeden arar, sıfır ağırlıktan başlar.
- **Pekiştirme rapor eyleminde.** Dron "bulundu" eylemiyle konum tahmini gönderir (v1'de fotoğraf/ASCII simüle edilmez, yalnızca konum ve bayrak). Tahmin doğru ve yeterince yakınsa (*benim çağrım:* 50 m içinde) ödül, yanlışsa ceza. Bunun dışında yalnızca pil maliyeti. Öğrenme durursa küçük bir keşif bonusu eklenebilir, varsayılan kapalı.
- **Kademe geçişi:** bulma oranı %60-70'e gelince. *Benim çağrım:* "gelince" = en az 200 ayrı tohumda %95 güven aralığının alt sınırı ≥ %60 ve baz çizgiden kötü değil. Aksi halde şansla ilerlenir.
- **Çoklu dron:** önce tek dron, sonra 2, sonra daha fazlası. Kardeşlere çarpmama için *benim çağrım:* minimum mesafe sert bir kural olur (güvenlik katmanı); öğrenilen kısım iş bölümü (aynı yeri taramamak). Haberleşme varsayımı (keşif maskesi paylaşılır mı, hangi menzilde) çoklu dron aşamasına gelince kararlaştırılır.
- **Ölçek:** tek pille mükemmel kapsama tavanı ~%72 (1 km), ~%18 (2 km), ~%8 (3 km). Geçiş kapısı anlamlı kalsın diye alan, dron sayısıyla birlikte büyür (örn. 2 dron ~1,4 km, 4 dron ~2 km).

## Kurucuya bırakılanlar

- RL kütüphanesi seçimi (kendi PPO'su mu, hazır kütüphane mi). Torch kurulumu bir sanal ortamda.
- Ödül ağırlıkları ve hücre boyutu gibi ayrıntılar.
- Hız için numpy vektörleştirme ya da numba.
- Eğitim sürecinin gece boyunca çökmeden çalışması (hafızadaki ilgili notu okuyarak).
