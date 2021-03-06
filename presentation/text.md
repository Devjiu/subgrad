Мы отправляемся от работы [1212.2002], в которой предложен оптимальный субградентный метод
на классе задач сильно выпуклой минимизации в предположении липщицевости целевого функционала.
min f(x), x \in Q,  

Метод будет выгледеть так:

x_{k+1} = proj_Q(x_k - h_k \nabla f(x_k) ) где h_k = 2  / (\mu (k+1)) 

Для него получаем оценку:

\label{eq:1}
f(\widehat(x)) - f(x_*) \leq 2 * M**2/ (\mu (N+1)) \forall x \in Q 

Заметим независимость оценки от удаленности старта от решения. 
Будем развивать данную идею в 2х направлениях - 
1) **Адаптивность** Уточнение оценки скорости сходимости путем замены константы глобальной константы на "усреднение" норм субградентов в точках итерационной последовательности.
2) **Относительная липщицевость**Обобщение результата на недавно возникший класс относительно липщицевых задач. Рассмотрение задач с функциональными огр., седловых задач и вариационных неравенств с относительно сильно монотонными и относительно ограниченными операторами. (пример - Лагранжева задача) \href{https://arxiv.org/abs/1710.04718}{[1710.04718]}, \href{https://proceedings.neurips.cc/paper/2020/file/b67fb3360ae5597d85a005153451dd4e-Paper.pdf}{[neur-ips 1]}, \href{https://arxiv.org/abs/2107.08011}{[2107.08011]}


Описываем Уточненную оценку. 
Показываем графики - говорим про целевую функции. 
1. Ограниченное Q - сравнение адаптивной и неадаптивной оценки 
2. Независимость от выбора точки старта
3. Пример на неограниченном Q - отсутвие условия Липщица. 

2я часть

Свойство относительной Липщицевости.
Сначала записываем для градиента - и упоминаем про V через дивергенцию Брегмана
[https://arxiv.org/pdf/1610.05708.pdf]

Относительно сильная выпуклость - для градиентов. 
(Привести примеры функций из статей ) лучше nips
Показываем еще раз оценку через M**2 - и говорим, что рассмотрим данный метод для задач с ограничениями.
Приводим пример функции Лагранжа. Описываем, что производная по ф. Лагранжа позволяет записать это в виде
вариационных неравенств. 

Вводим понятие относительной монотонности - приводим пример, который показывает, что
относительная монотонность для оператора порождает \mu-относительную сильную выпуклость для f() в случае потенциального оператора g()

Формулируем теорему для оценки с M**2.

Следствие с количеством итераций.

Змечание для сильно выпуклой прокс-функции (модификация \mu-относительной монотонности)

Говорим про относительно сильно выпукло-вогнутые седловые задачи 
И пишем оценку для относительно сильно монотонных ВН с относительно ограниченным оператором
Можно сказать про оптимальность подхода для асимптотики. (Оценка в терминах O больших совпадает нижней оценкой.)



Занимаюсь реализацией подходов сравниваемых в статье для статьи мат. заметки, 
чтобы воспроизвести полученные там экспериментальные данные.
Столкнулся с проблемой при решении вспомогательной задачи на argmin - у меня получилась достаточно
громоздкая конструкция, чтобы ее решать на каждом шаге. Кажется, что можно упростить.
Речь идет об обоих видах зеркального спуска - рассматривается подзадача \argmin_{x \in Q} <g(x_k, x)> + V(x, x_k).
Вот ее нужно записать в явном виде для x при d(x) = \norm(x) ^ {2p} / 2p (прокс-функция).
Для повторения теста нужно p = 2, но свойства относительной выпуклости проявляются начиная с p = 4,
что тоже стоит проверить.
Я написал о своей проблеме Федору Сергеевичу - он сказал, что более простая запись уровнения - 
\nabla d(x_{k+1}) = \nabla d(x_k) - h_k g(x_k) и стоит посмотреть в статье 1610.05708 подробнее.
Изначально я расписал условие минимума градиента вспомогательной функции явно, но получилось, как мне кажется,
громоздко.
Сейчас попробую использовать подход, описанный в статье, чтобы получить более явную форму записи шага.  
