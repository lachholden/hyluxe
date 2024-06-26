(import decimal [Decimal])
(import math)
(import dataclasses [dataclass])

(require hyrule.argmove [->])
(require ptf.util :readers [D])

(import matplotlib.pyplot)

(setv
  MINS "\u2032"
  SECS "\u2033")

  (Decimal beeepj)

(defreader D
  (.slurp-space &reader)
  (assert  (.peek-and-getc &reader "\""))
  (let [decimal-chars-list
        (takewhile (fn [c] (!= "\"" c)) (.chars &reader))

        decimal-string
        (.join "" decimal-chars-list)]

    `(Decimal ~decimal-string)))

(defclass [(dataclass :frozen True)] Pace []
  "Represents a pace per some unit distance (i.e. typically time per km.)

  Minimum resolution is seconds, and can support arbitrary fractional-second
  resolution.

  Stored internally as a Decimal quantity of seconds.
  "

  #^ Decimal seconds

  (defn [classmethod] from-seconds [cls seconds]
    (cls (Decimal seconds)))

  (defn __str__ [self]
    (let [mm (math.floor (/ self.seconds 60))
          ss (.quantize (- self.seconds (* mm 60)) #D(Decimal))
          ff (- self.seconds ss (* mm 60))]
      f"{mm}{MINS}{ss :02.0f}{(-> ff str (.lstrip "0"))}{SECS}")))


#_(
  (Pace.from-seconds "120.5"))
