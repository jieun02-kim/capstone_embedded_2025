(define (domain follow)

    (:requirements :typing :adl)

    (:types
        patient
    )

    (:predicates
        (following ?p - patient)
        (visible ?p - patient)
    )

    (:action follow
        :parameters (?p - patient)
        :precondition (and
            (visible ?p)
        )
        :effect (and
            (following ?p)
        )
    )
)