(ns jsa.genseq
  (:require [tech.v3.dataset :as ds]
            [tech.v3.tensor :as dtt]
            [tech.v3.parallel.for :as pfor]
            [aerial.bio.utils.aligners :as aln]
            [aerial.utils.string :as str]
            [aerial.utils.misc :as aum]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.bitmap :as bitmap])
  (:import [tech.v3.datatype NDBuffer]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(def src-sequences
  ["GTCGTGACGTCGTGCGTAGCGGGAGGGCGGTGCGTGTGCCCTC" "GCGCACAACGCCGAGTACAATGTACCGCGTTACGATGCGTC"
   "ACGTGCACACGGTGACGGCAAAGCATCGCCCCACGGTTGCGTC" "GCGCACAACGCCGAGTACAATGTACCGCGTTACGATGCGTC"
   "ACGGCAGCGCCACGACACGTGCGTCACGGTGCACCGTCACGTC" "GCGCACAACGCCGAGTACAATGTACCGCGTTACGATGCGTC"
   "TTCCGTCGTGACGTCGTGCGTAGCGAACAGCTTACGATGCGTC" "GCGCACAACGCCGAGTACAATGTACCGCGTTACGATGCGTC"
   "TGCATCACGACGGCACCTAAGACGTCCATGGCGTGCGTCACGC" "GCGCACAACGCCGAGTACAATGTACCGCGTTACGATGCGTC"
   "ACACGGCAGCATCATGGCACGTGCGTCACGGTGACGGCCAAGC" "GCGCACAACGCCGAGTACAATGTACCGCGTTACGATGCGTC"
   "ACGGCAGCGCCACGACACGTGCGTCACGGTGACGGCCAAGCA" "GCGCACAACGCCGAGTACAATGTACCGCGTTACGATGCGTC"])


(defn sequences
  [n]
  (repeatedly n #(src-sequences (rand-int (count src-sequences)))))


(def src-ds (ds/->dataset {:a (sequences 3000)
                           :b (sequences 3000)}))


(defn roundit [r & {:keys [places] :or {places 4}}]
  (let [n (Math/pow 10.0 (double places))]
    (-> r double (* n) Math/round (/ n))))


(defmacro deep-aget
  ([hint array idx]
    `(aget ~(vary-meta array assoc :tag hint) ~idx))
  ([hint array idx & idxs]
    `(let [a# (aget ~(vary-meta array assoc :tag 'objects) ~idx)]
       (deep-aget ~hint a# ~@idxs))))


(defmacro deep-aset [hint array & idxsv]
  (let [hints '{doubles double, longs long, ints int}
                ;;vectors clojure.lang.PersistentVector}
        [v idx & sxdi] (reverse idxsv)
        idxs (reverse sxdi)
        v (if-let [h (hints hint)] (list h v) v)
        nested-array (if (seq idxs)
                       `(deep-aget ~'objects ~array ~@idxs)
                        array)
        a-sym (with-meta (gensym "a") {:tag hint})]
      `(let [~a-sym ~nested-array]
         (aset ~a-sym ~idx ~v))))



(defn- init-matrix
  "We use Java arrays. Which means we are in mutation land, which we
  trade for speed. Also, the arrays are strictly contained within a
  single alignment computation (each alignment allocates their own
  array)."
  [rows cols endsgap]
  (let[scmat (make-array (class []) rows cols)
       endsgap (long endsgap)]
    (dotimes [i (long rows)]
      (dotimes [j (long cols)]
        (deep-aset objects scmat i j
              (cond (= i j 0) [:- 0 [0 0]]
                    (= i 0) [:l (* j endsgap) [0 j]]
                    (= j 0) [:u (* i endsgap) [i 0]]
                    :else []))))
    scmat))

(defn- init-tensor
  ^NDBuffer [rows cols endsgap]

  ;;Fastest but not by much and very noisy
  #_(let [endsgap (long endsgap)
        rows (long rows)
        cols (long cols)
        n-elems (* rows cols)
        data (object-array n-elems)]
    (pfor/parallel-for
     idx n-elems
     (let [i (quot idx cols)
           j (rem idx cols)]
       (aset data idx
             (cond (== i j 0) [:- 0 [0 0]]
                   (== i 0) [:l (* j endsgap) [0 j]]
                   (== j 0) [:u (* i endsgap) [i 0]]
                   :else []))))
    (dtt/reshape data [rows cols]))
  ;;Slightly slower and much cleaner
  (let [endsgap (long endsgap)]
    (-> (dtt/compute-tensor [rows cols]
                            (fn [^long i ^long j]
                              (cond (== i j 0) [:- 0 [0 0]]
                                    (== i 0) [:l (* j endsgap) [0 j]]
                                    (== j 0) [:u (* i endsgap) [i 0]]
                                    :else [])))
        (dtype/clone))))

(defn- score
  "Compute and return the best score and its direction :l for 'from
  left' :u for 'from up' and :d for 'from diagonal'. There is a lot of
  ugly stuff here to keep cpu cycles low. For example, we don't need
  to use an explicit substitution matrix and can use match and
  m(is)match scores directly (a 10x speed up) and we don't explicitly
  sort the resulting scores instead opting for a nested (if .. then
  else) approach. And a lot of primitive typing to ensure unboxed
  arithmatic. Also use of str/get removes chatAt reflection!! (or could
  have added ^String hints to s1 and s2, but ugly)"
  [i j c1 c2 scmat kind, gap match mmatch submat]
  ;;(println :I i :J j (map vec (vec scmat)))
  (let [i (long i)
        j (long j)
        gap (long gap)
        match (long match)
        mmatch (long mmatch)
        ;;c1 (str/get s1 i)
        ;;c2 (str/get s2 j)
        submat? submat
        lsc (+ (long ((deep-aget objects scmat i (dec j)) 1)) gap)
        usc (+ (long ((deep-aget objects scmat (dec i) j) 1)) gap)
        dsc (+ (long ((deep-aget objects scmat (dec i) (dec j)) 1))
               (if submat?
                 (long (submat [c1 c2]))
                 (if (= c1 c2) match mmatch)))
        score (if (> lsc usc)
                (if (> lsc dsc)
                  [:l lsc]
                  [:d dsc])
                (if (> usc dsc)
                  [:u usc]
                  [:d dsc]))
        ;; Check if local for possible start location of [i j]
        score (if (and (= kind :local) (> 0 (long (score 1)))) [:s 0] score)]
    (conj score [i j])))


(defn- score-t
  "Compute and return the best score and its direction :l for 'from
  left' :u for 'from up' and :d for 'from diagonal'. There is a lot of
  ugly stuff here to keep cpu cycles low. For example, we don't need
  to use an explicit substitution matrix and can use match and
  m(is)match scores directly (a 10x speed up) and we don't explicitly
  sort the resulting scores instead opting for a nested (if .. then
  else) approach. And a lot of primitive typing to ensure unboxed
  arithmatic. Also use of str/get removes chatAt reflection!! (or could
  have added ^String hints to s1 and s2, but ugly)"
  [i j c1 c2 scmat kind gap match mmatch submat]
  ;;(println :I i :J j (map vec (vec scmat)))
  (let [i (long i)
        j (long j)
        gap (long gap)
        match (long match)
        mmatch (long mmatch)
        ;;c1 (str/get s1 i)
        ;;c2 (str/get s2 j)
        c1 (char c1)
        c2 (char c2)
        submat? submat
        ^NDBuffer scmat scmat
        lsc (+ (long ((.ndReadObject scmat i (dec j)) 1)) gap)
        usc (+ (long ((.ndReadObject scmat (dec i) j) 1)) gap)
        dsc (+ (long ((.ndReadObject scmat (dec i) (dec j)) 1))
               (if submat?
                 (long (submat [c1 c2]))
                 (if (== 0 (Character/compare c1 c2)) match mmatch)))
        score (if (> lsc usc)
                (if (> lsc dsc)
                  [:l lsc]
                  [:d dsc])
                (if (> usc dsc)
                  [:u usc]
                  [:d dsc]))
        ;; Check if local for possible start location of [i j]
        score (if (and (identical? kind :local)
                       (> 0 (long (score 1))))
                [:s 0]
                score)]
    (conj score [i j])))


(defmacro do-col [scmat col e & body]
  `(dotimes [r# (alength ~scmat)]
     (let [~e (deep-aget ~'objects ~scmat r# ~col)]
       ~@body)))

(defmacro do-row [scmat row e & body]
  `(let [scr# ^"[Lclojure.lang.PersistentVector;" (aget ~scmat 0)]
     (dotimes [c# (alength scr#)]
       (let [~e (deep-aget ~'objects ~scmat ~row c#)]
         ~@body))))


(defn- find-start
  "Find the starting cell in the score matrix for trace back. This is
  also the cell with the 'best' score for each KIND of alignment:

  :global - Standard NW global alignment
  :ends-gap-free - global alignment without gaps on ends (prefix/suffix aln)
  :local - Standard SW local alignment

  This also contains a lot of ugly low level matrix access code to
  avoid consing and vector overhead. Made somewhat better by do-col
  and do-row macros.
  "
  [^"[[Lclojure.lang.PersistentVector;" scmat, rows cols kind]
  (case kind
    :global (aget scmat rows cols)
    :ends-gap-free
    (let [max (volatile! (aget scmat rows cols))]
      (do-row scmat rows v
              (when (> (long (v 1)) (long (@max 1))) (vswap! max (fn[_] v))))
      (do-col scmat cols v
              (when (> (long (v 1)) (long (@max 1))) (vswap! max (fn[_] v))))
      @max)
    :local
    (let [max (volatile! (aget scmat rows cols))]
      (dotimes [r rows]
        (do-row scmat r v
                (when (> (long (v 1)) (long (@max 1))) (vswap! max (fn[_] v)))))
      @max)))


(defmacro do-col-t [scmat col e & body]
  `(let [rows# (dtt/rows ~scmat)]
     (dotimes [r# (dtype/ecount rows#)]
       (let [~e (rows# ~col)]
         ~@body))))

(defmacro do-row-t [scmat row e & body]
  `(let [cols# (dtt/columns ~scmat)]
     (dotimes [c# (dtype/ecount cols#)]
       (let [~e (cols# ~row)]
         ~@body))))


(defn- find-start-t
  [^NDBuffer scmat rows cols kind]
  (case kind
    :global (.ndReadObject scmat rows cols)
    :ends-gap-free
    (let [max (volatile! (.ndReadObject scmat rows cols))]
      (do-row-t scmat rows v
              (when (> (long (v 1)) (long (@max 1))) (vswap! max (fn[_] v))))
      (do-col-t scmat cols v
              (when (> (long (v 1)) (long (@max 1))) (vswap! max (fn[_] v))))
      @max)
    :local
    (let [max (volatile! (.ndReadObject scmat rows cols))]
      (dotimes [r rows]
        (do-row-t scmat r v
                  (when (> (long (v 1)) (long (@max 1))) (vswap! max (fn[_] v)))))
      @max)))


(defn- trace-back [scmat rows cols kind]
  (let [start (find-start scmat rows cols kind)]
    #_(clojure.pprint/pprint (map vec (vec scmat)))
    (loop [P (list start)
           [r c] (start 2)]
      #_(println r c P)
      (cond
        (and (= kind :global) (= r c 0)) (rest P)
        (and (not= kind :global) (= ((aget scmat r c) 1) 0)) (rest P)
        :else
        (let [r (long r)
              c (long c)
              cell (aget scmat r c)
              dir (cell 0)
              step (case dir
                     :d (deep-aget objects scmat (dec r) (dec c))
                     :u (deep-aget objects scmat (dec r) c)
                     :l (deep-aget objects scmat r (dec c))
                     (aum/raise
                      :dash-dir? "Bad direction"
                      :cell cell :r r :c c))]
          (recur (conj P step) (step 2)))))))


(defn- trace-back-t [^NDBuffer scmat rows cols kind]
  (let [start (find-start-t scmat rows cols kind)]
    #_(clojure.pprint/pprint (map vec (vec scmat)))
    (loop [P (list start)
           [r c] (start 2)]
      #_(println r c P)
      (cond
        (and (= kind :global) (= r c 0)) (rest P)
        (and (not= kind :global) (= ((.ndReadObject scmat r c) 1) 0)) (rest P)
        :else
        (let [r (long r)
              c (long c)
              cell (.ndReadObject scmat r c)
              dir (cell 0)
              step (case dir
                     :d (.ndReadObject scmat (dec r) (dec c))
                     :u (.ndReadObject scmat (dec r) c)
                     :l (.ndReadObject scmat r (dec c))
                     (aum/raise
                      :dash-dir? "Bad direction"
                      :cell cell :r r :c c))]
          (recur (conj P step) (step 2)))))))


(defn decode-trace-back
  [s1 s2 tbk]
  (let [first-cell (first tbk)
        last-cell (last tbk)
        scr (second last-cell)
        start-idx (->> first-cell last (mapv dec))
        last-idx  (->> last-cell last (mapv dec))
        char-pair (fn[[dir scr [i j]]]
                    (case dir
                      :d [(str/get s1 i) (str/get s2 j)]
                      :l ["-" (str/get s2 j)]
                      :u [(str/get s1 i) "-"]))
        pairs (map char-pair tbk)]
    [[scr start-idx last-idx]
     [(apply str (map first pairs))
      (apply str (map second pairs))]]))


(defn align
  "Align s1 and s2 (strings) in the manner defined by kind:

  :global - Standard NW global alignment
  :ends-gap-free - global aln without gaps on ends (prefix/suffix aln)
  :local - Standard SW alignment

  submat is a substitution map (matrix) giving a score for a character
  pair. This namespace provides nt4submat which encodes a 'standard'
  ATGCxATGC substitution matrix.

  match is a score for the case when characters match (they are =)
  mmatch is a penalty score for the case when characters do not match

  submat and match/mmmatch are mutually exclusive

  gap is a penalty score for indels (characters match a gap)

  keep-scmat is a boolean for whether to return the scoring
  matrix (true) or not (false)

  Returns the traceback path of best alignment as a vector of
  triples: [dir score [i j]], where dir is the direction to get this
  point of path (:d -> diagonal, :u -> up, :l -> left), score is the
  score at this point, and [i j] is the index pair for s1 and s2 of
  this point (i for s1, j for s2) 0-based.

  If decode is true (default), tb is returned as the pair of aligned
  strings (with gaps if needed)

  If keep-scmat is true, returns a vector [tb scmat] otherwise returns
  just tb, tb the traceback.
  "
  [s1 s2 & {:keys [kind submat gap match mmatch decode keep-scmat]
            :or {kind :global gap -2 decode true keep-scmat false}}]
  (let [rows (count s1)
        cols (count s2)
        s1 (str "-" s1)
        s2 (str "-" s2)
        endsgap (if (= kind :global) gap 0)
        scmat (init-matrix (inc rows) (inc cols) endsgap)]
    (dotimes [c (long cols)]
      (dotimes [r (long rows)]
        (let [c (inc c)
              r (inc r)
              ch1 (str/get s1 r)
              ch2 (str/get s2 c)]
          (deep-aset objects scmat r c
                     (score r c ch1 ch2 scmat kind gap match mmatch submat)))))
    (let [tb (trace-back scmat rows cols kind)
          tb (if decode (decode-trace-back s1 s2 tb) tb)]
      (if keep-scmat [tb scmat] tb))))


(defn align-t
  [s1 s2 & {:keys [kind submat gap match mmatch decode keep-scmat]
            :or {kind :global gap -2 decode true keep-scmat false}}]
  (let [rows (count s1)
        cols (count s2)
        s1 (str "-" s1)
        s2 (str "-" s2)
        endsgap (if (identical? kind :global) gap 0)
        scmat (init-tensor (inc rows) (inc cols) endsgap)]
    (dotimes [c (long cols)]
      (dotimes [r (long rows)]
        (let [c (inc c)
              r (inc r)
              ch1 (str/get s1 r)
              ch2 (str/get s2 c)]
          (.ndWriteObject scmat r c
                          (score-t r c ch1 ch2 scmat kind gap match mmatch submat)))))
    (let [tb (trace-back-t scmat rows cols kind)
          tb (if decode (decode-trace-back s1 s2 tb) tb)]
      (if keep-scmat [tb scmat] tb))))





(defrecord AlnhamRecord [^double score ^double chdiff ^double maxlen ^double %same])


(defn alnham
  "'glorified' (and expensive!) hamming using full global alignment with
  0 penalties for misqmatch and indels and 1 for match.  So, basically
  counts up the total matching bases in s1 and s2 (both sequences as strings)

  Returns a 'score' map with keys
  :score (count of matching bases)
  :chdiff (difference between max string size and score)
  :%same (percentage of matching base count to max size)
  :maxlen (max length of s1 and s2)"
  ^AlnhamRecord [s1 s2]
  (let [[[score _ lens]] (align s1 s2 :match 1, :mmatch 0, :gap 0)
        maxlen (->> lens (apply max) double inc)
        %same (-> score double (/ maxlen) roundit)]
    (AlnhamRecord. score (- maxlen (double score)) maxlen %same)))


(defn alnham-t
  "'glorified' (and expensive!) hamming using full global alignment with
  0 penalties for misqmatch and indels and 1 for match.  So, basically
  counts up the total matching bases in s1 and s2 (both sequences as strings)

  Returns a 'score' map with keys
  :score (count of matching bases)
  :chdiff (difference between max string size and score)
  :%same (percentage of matching base count to max size)
  :maxlen (max length of s1 and s2)"
  ^AlnhamRecord [s1 s2]
  (let [[[score _ lens]] (align-t s1 s2 :match 1, :mmatch 0, :gap 0)
        maxlen (->> lens (apply max) double inc)
        %same (-> score double (/ maxlen) roundit)]
    (AlnhamRecord. score (- maxlen (double score)) maxlen %same)))


(defn alnham-same
  ^double [s1 s2]
  (let [[[score _ lens]] (align s1 s2 :match 1 :mmatch 0 :gap 0)
        maxlen (->> lens (apply max) double inc)]
    (-> score double (/ maxlen) roundit)))


(defn add-score-cols [ds]
  (ds/column-map
   ds :%same
   alnham-same
   :float32
   [:a :b]))


(defn add-score-cols-untyped
  [ds]
  (ds/column-map
   ds :%same
   alnham-same
   [:a :b]))


(defn add-score-cols-nonlazy
  [ds]
  (assoc ds :%same
         (-> (dtype/emap alnham-same :float32 (ds :a) (ds :b))
             (dtype/clone))))


(defn add-score-rowmap
  [ds]
  (ds/row-map ds #(alnham (% :a) (% :b))))


(defn add-score-rowmap-t
  [ds]
  (ds/row-map ds #(alnham-t (% :a) (% :b))))


(defn run-algo
  [add-score-fn]
  (-> src-ds
      (add-score-fn)
      (ds/sort-by-column :%same >)))


(comment
  (def align-res (align (src-sequences 0) (src-sequences 1) :match 1, :mmatch 0, :gap 0))
  (def align-res-t (align-t (src-sequences 0) (src-sequences 1) :match 1, :mmatch 0, :gap 0))
  (def result (alnham (src-sequences 0) (src-sequences 1)))
  (def result-t (alnham-t (src-sequences 0) (src-sequences 1)))
  (def correct-result {:score 25.0, :chdiff 18.0, :maxlen 43.0, :%same 0.5814})

  (time (def ignored (run-algo add-score-cols)))
  ;; 440ms
  (time (def ignored (run-algo add-score-cols-untyped)))
  ;; 2688ms
  (time (def ignored (run-algo add-score-cols-nonlazy)))
  ;;  436

  (time (def ignored (run-algo add-score-rowmap)))
  ;;  445ms

  (crit/quick-bench (run-algo add-score-rowmap))
;; Evaluation count : 6 in 6 samples of 1 calls.
;;              Execution time mean : 449.256405 ms
;;     Execution time std-deviation : 6.713039 ms
;;    Execution time lower quantile : 444.955685 ms ( 2.5%)
;;    Execution time upper quantile : 460.637486 ms (97.5%)
;;                    Overhead used : 1.732553 ns

;; Found 1 outliers in 6 samples (16.6667 %)
;; 	low-severe	 1 (16.6667 %)
;;  Variance from outliers : 13.8889 % Variance is moderately inflated by outliers
  nil

  (crit/quick-bench (run-algo add-score-rowmap-t))
;; Evaluation count : 6 in 6 samples of 1 calls.
;;              Execution time mean : 189.151433 ms
;;     Execution time std-deviation : 2.154052 ms
;;    Execution time lower quantile : 187.459430 ms ( 2.5%)
;;    Execution time upper quantile : 192.568238 ms (97.5%)
;;                    Overhead used : 1.732553 ns

  (time (def ignored (run-algo add-score-rowmap-t)))
  ;;
  ;;Profile run
  (dotimes [idx 1000]
    (run-algo add-score-rowmap-t))

  )
