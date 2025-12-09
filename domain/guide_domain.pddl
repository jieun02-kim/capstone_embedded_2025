(define (domain guide)

    (:requirements :typing :adl)

    (:types
        location
    )

    (:predicates
        (at ?loc - location)
        (destination ?loc - location)
        (visible ?loc - location)
    )

    (:action go
        :parameters (?loc - location)
        :precondition (and
            (destination ?loc)
            (visible ?loc)
        )
        :effect (and
            (at ?loc)
        )
    )
)