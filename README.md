# 💊 İlaç/Hap Sınıflandırma ve Optimizer Analizi (CNN)

Bu proje, görüntü işleme (Computer Vision) ve Derin Öğrenme (Deep Learning) teknikleri kullanılarak ilaç ve hap görüntülerini sınıflandırmak amacıyla geliştirilmiştir. Projenin temel odak noktası, bir Evrişimli Sinir Ağı (CNN) mimarisi üzerinde farklı optimizasyon algoritmalarının (Adam, RMSProp ve Momentumlu SGD) performans ve yakınsama (convergence) hızlarını karşılaştırmaktır.

## 🚀 Proje Detayları
Endüstri standartlarında bir CNN modeli kurularak, toplamda 20.000 görüntüden oluşan veri seti üzerinde 10 farklı ilaç sınıfı için eğitim gerçekleştirilmiştir. 

**Karşılaştırılan Optimizer'lar:**
* `Adam` (Hızlı ve adaptif öğrenme)
* `RMSprop` (Dalgalı verilerde kararlı ilerleme)
* `SGD + Momentum` (Derin mimarilerde klasik ve güvenilir yaklaşım)

## 📊 Veri Seti ve Ön İşleme
* **Veri Sayısı:** 20.000 görüntü
* **Dağılım:** %80 Eğitim (16.000) / %20 Doğrulama (4.000)
* **Sınıf Sayısı:** 10
* **Görüntü Boyutu:** 128x128 piksel
* **Normalizasyon:** 0-1 aralığında ölçeklendirme (Rescaling)

## 🛠️ Kullanılan Teknolojiler
* Python 3.x
* TensorFlow / Keras
* Matplotlib, NumPy, OS

## ⚙️ Kurulum ve Çalıştırma
Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. Repoyu bilgisayarınıza klonlayın:
   ```bash
   git clone [https://github.com/Berrinsrc/Pill-Classification-Analysis.git](https://github.com/Berrinsrc/Pill-Classification-Analysis.git)