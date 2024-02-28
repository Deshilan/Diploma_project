# Analiza postoju pojazdów w miejskiej przestrzeni parkingowej

Aplikacja była kluczową częścią mojej pracy dyplomowej. Jej główne cele to:
-określanie zajętości miejsc,
-odnajdywanie niepoprawnie zaparkowanych pojazdów i określanie, czy któryś z nich jest uprzywilejowany (w takim wypadku
program miał ignorować błąd).
Program miał opierać swoje działanie o uczenie maszynowe. Praca miała charakter eksperymentalny: oprócz samego
stworzenia działającego programu dążono do tego, by osiągał on jak najlepsze wyniki, zarówno pod kątem dokładności, jak
i czasu działania.


## Dokumentacja oraz wyniki badań
Wszystkie informacje wymagane do zrozumienia działania programu oraz mojego podejścia a także wyniki uzyskane przez
poszczególne modele znajdują się w pliku JW_Praca_dyplomowa.pdf. Należy zaznaczyć, że część mnej istotnych wyników, w
szczególności tych uzyskanych przez nieefektywne modele, nie znalazła się w pracy. Nie mniej pozostałości po nich nadal
znajdują się w plikach- mimo, że okazały się nieprzydatne, można z nich wysnuć pewne wnioski.


## Wykorzystane technologie
Wszystkie niezbędne biblioteki (w niektórych przypadkach najpewniej także archiwalne, niekoniecznie niezbędne w finalnej
wersji) są wymienione w pliku requirements.txt.

Najważniejsze z wykorzystanych technologii/bibliotek to:
-Python 3 (wraz z szeregiem podstawowych bibliotek, np. NumPy),
-Tensorflow,
-PyTorch,
-PyCharm,
-CUDA/cuDNN (pozwalają korzystać z GPU).

Dodatkowo, w celu powtórzenia prowadzonych eksperymentów, potrzebne mogą być zbiory danych. Z uwagi na rozmiar
udostępniam je na dysku Google.
LINK: https://drive.google.com/drive/folders/12nre4hY4b9b-cYI2FM9p63tYv1yR33b1?usp=drive_link


## Uruchamianie programu
W obecnej wersji projekt ma charakter poglądowy. Możliwe jest jego uruchomienie po podmianie części ścieżek. Obecnie
trwają prace mające na celu stworzenie testowej, łatwej do uruchomienia wersji, korzystającej zarówno z CPU jak i GPU
(w pewnym stopniu stworzenie takiego oprogramowania utrudnia stosowanie przez autora systemu Windows, który nie wspiera
wykorzystania GPU). Finalna wersja powinna pojawić się jeszcze w lutym.

Same przykładowe efekty działania programu znajdują się w folderze RESULTS.


## W jaki sposób projekt pozwolił mi się rozwinąć?
Z mojego punktu widzenia był to pierwszy poważniejszy projekt związany z uczeniem maszynowym. Chociaż można poddawać w
wątpliwość jego komercyjność, był on ważny z punktu widzenia merytorycznego: skłonił mnie nie tylko do poszerzenia wiedzy
w zakresie teoretycznym (chociaż AI byłem zainteresowany już od dłuższego czasu), ale także pozwolił na rozwój w sferze
praktycznej: w końcu musiałem zapoznać się z przynajmniej podstawami kluczowych bibliotek Python'a związanych ze sztuczną
inteligencją: Tensorflow'em i PyTorch'em.