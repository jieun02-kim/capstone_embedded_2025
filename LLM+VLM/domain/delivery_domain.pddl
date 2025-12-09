(define (domain delivery)

  (:requirements :typing :adl)

  (:types
      location door patient
  )

  (:predicates
      (at ?loc - location)
      (destination ?loc - location)
      (visible ?obj - (door patient))
      (has_label ?d - door ?loc - location)
      (greeted ?t - patient)
      (delivered ?loc - location)
  )

  (:action greet
      :parameters (?t - patient)
      :precondition (and
          (visible ?t)
      )
      :effect (and
          (greeted ?t)
      )
  )

  (:action arrive
      :parameters (?loc - location ?d - door)
      :precondition (and
          (visible ?d)
          (has_label ?d ?loc)
      )
      :effect (and
          (at ?loc)
          (arrived ?loc)
      )
  )
)