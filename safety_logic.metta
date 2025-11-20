; ============================================================
;  HYPERON SAFETY KNOWLEDGE BASE (MeTTa)
;  The Symbolic Logic Layer for the Artificial Pancreas
; ============================================================

; 1. Define Safety Thresholds (Knowledge Atoms)
(= (SafeLow) 70)
(= (SafeIOB) 8.0)
(= (MaxUncertainty) 50)

; 2. Define Action Modifications
; Action 4 = Stop Insulin (TEMP_BASAL 0%)
; Action 2 = Medium Bolus (Fallback from High Bolus)
; Action 0 = Do Nothing

; 3. The Core Logic Function
; Syntax: (evaluate-safety $action_id $cgm $iob $uncertainty)

; Rule A: HYPOGLYCEMIA BLOCK
; If Glucose < 70, force Action 4 (Stop Insulin) regardless of what the NN wants.
(= (evaluate-safety $act $cgm $iob $uncert)
   (if (< $cgm (SafeLow))
       (Result 4 "CRITICAL: Glucose below safety threshold. Insulin Suspended.")
       (check-iob $act $cgm $iob $uncert)))

; Rule B: IOB SATURATION BLOCK
; If Insulin on Board > 8.0 units, block any Bolus (Actions 1, 2, 3).
(= (check-iob $act $cgm $iob $uncert)
   (if (and (> $iob (SafeIOB)) (> $act 0))
       (Result 0 "SAFETY: Active Insulin too high. Bolus blocked.")
       (check-uncertainty $act $cgm $iob $uncert)))

; Rule C: UNCERTAINTY DAMPENING
; If the AI is confused (Uncertainty > 50) and tries a large bolus (3), reduce to medium (2).
(= (check-uncertainty $act $cgm $iob $uncert)
   (if (and (> $uncert (MaxUncertainty)) (== $act 3))
       (Result 2 "UNCERTAINTY: Neural confidence low. Aggressive bolus reduced.")
       (Result $act "Action Verified by Hyperon.")))