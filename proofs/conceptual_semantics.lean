import Mathlib.Data.Set.Basic
open Set

axiom Entity : Type

/-
            ***  2.1 Definitions ***
-/

/-
    2.1.1 Concept
-/

-- define predicate $isEntity$ that yields a true proposition when applied to any entity $x$
axiom isEntity : Entity → Prop

-- $R$ is the Set of all entities
axiom R : Set Entity
axiom isEntityInR : ∀x, isEntity x → x ∈ R

-- show that an arbitrary entity $e$ is in $R$
example (h1: isEntity e) : e ∈ R := by
    exact isEntityInR e h1

-- selection criterion is a predicate that can be applied to entities (of type Entity → Prop)

-- an arbitrary concept $C$ is a set of **selection criteria**
variable (C : Set (Entity → Prop))

/-
1. $isReferentOfC$ is a function that yields a true proposition for any entity $x$ that is a referent of concept $C$
   i.e. for any entity $x$, if $x$ is a referent of concept $C$, then for all predicates (selection criteria) $p$ in $C$ the proposition $p(x)$ is true
-/
def isReferentOfC (x : Entity) : Prop := ∀ p ∈ C, p x


/-
    2.1.2 Semantic field
-/

/-
2. The set of referents $R_C$ of an arbitrary concept $C$ is a collection of entities that satisfy all predicates (selection criteria) in $C$
   Subsequently, we will refer to $R_C$​ as the **semantic field** of concept $C$.
-/
def R_C : Set Entity := {x | isReferentOfC C x}

/-
    2.1.4 Equivalence of selection criteria
-/

/-
4. Selection criteria $a$ and $b$ are equivalent if and only if they imply the same set of referents
-/
def equivalentSelectionCriteria (a b : Entity → Prop) : Prop := ∀ x: Entity, a x ↔ b x

/-
    2.1.7 Conceptual equivalence
-/

/-
7. Concepts $C$ and $D$ are equivalent if and only if they have the same semantic field
-/
def equivalentConcepts (C D : Set (Entity → Prop)) : Prop := R_C C = R_C D

/-
            ***  2.2 Properties ***
-/

/-
    2.2.1 Compositionality of concepts and noun phrases
-/

/-
8. Adding a selection criterion $c$ to concept $C$ yields a concept $C' = C ∪ {c}$ such that the semantic field of $C'$ is a subset of the semantic field of $C$
-/
theorem additiveCompositionalityForward (c : Entity → Prop) : R_C (C ∪ {c}) ⊆ R_C C := by
    intro x hx
    intros p hp
    apply hx
    simp [hp]

/-
9. Removing a selection criterion $c$ from concept $C$ yields a concept $C' = C \ {c}$ such that the semantic field of $C$ is a subset of the semantic field of $C'$
-/
theorem additiveCompositionalityBackward (c : Entity → Prop) : R_C C ⊆ R_C (C \ {c})  := by
    intro x hx
    intros p hp
    apply hx
    exact hp.left

/-
    2.2.2 Compositionality of meaning
-/

/-
10. Contradictory selection criteria $a$ and $b$ result in a concept with an empty semantic field.
    Such concepts can be regarded as nonsensical or meaningless (as opposed to meaningful), because there is no context where they can have referents.
-/
theorem meaninglessConcept (a b : Entity → Prop) (hx: ∀ x: Entity, ¬(a x ∧ b x)) (ha: a ∈ C) (hb: b ∈ C): R_C C = ∅ := by
    ext x
    constructor
    · intro h
      have h1 := h a ha
      have h2 := h b hb
      exact hx x ⟨h1, h2⟩
    · intro h
      exact False.elim h


/-
    2.2.3 Compositionality of logical operations in natural language
-/

/-
11. Combination of concepts with "or" implies the union of their semantic fields.
-/
theorem compositionalityofLogicOr (C_i C_j : Set (Entity → Prop)) (r : Entity) : (r ∈ (R_C C_i)) ∨ (r ∈ (R_C C_j)) → r ∈ ((R_C C_i) ∪ (R_C C_j)) := by
  intro h
  cases h with
  | inl h => exact Or.inl h
  | inr h => exact Or.inr h

/-
12. Combination of concepts with "and" implies an intersection of their semantic fields
-/
theorem compositionalityofLogicAnd (C_i C_j : Set (Entity → Prop)) (r : Entity) : (r ∈ (R_C C_i)) ∧ (r ∈ (R_C C_j)) → r ∈ ((R_C C_i) ∩ (R_C C_j)) := by
  intro h
  exact And.intro h.left h.right
