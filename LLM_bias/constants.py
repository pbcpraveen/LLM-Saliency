from enum import Enum

ENTITY = "entity"
CONTEXTUALISING_ATTRIBUTES = "contextualising_attributes"
TARGET_ATTRIBUTES = "target_attributes"

# Sources
WIKIBIO = "wiki_bio"
NOBEL_PRIZE_DATASET = ("https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/nobel-prize-laureates/exports/"
                       "json?lang=en&timezone=America%2FLos_Angeles")
NOBEL_PRIZE = "nobel_prize"
MOVIE_DATASET = ("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
MOVIE = "movie"

COUNTRY_DATASET = ("https://raw.githubusercontent.com/bastianherre/global-leader-ideologies/"
                   "main/global_leader_ideologies.csv")

class ConceptClass(Enum):
    PLACE = "place"
    PERSON_NAME = "name"
    YEAR = "year"


class EntityClass(Enum):
    PERSON = "person"
    NOBEL_LAUREATES = "nobel_laureates"
    MOVIE = "movie"


class Attribute(Enum):
    NAME = "name"
    NATIONALITY = "nationality"
    OCCUPATION = "occupation"
    BIRTH_DATE = "birth_date"
    DEATH_DATE = "death_date"
    BIRTH_PLACE = "birth_place"
    DEATH_PLACE = "death_place"
    MOTIVATION_NOBEL = "motivation"
    CATEGORY_NOBEL = "category"
    BIRTH_DATE_NOBEL = "born"
    DEATH_DATE_NOBEL = "died"
    YEAR = "year"
    BIRTH_CITY = "borncity"
    DEATH_CITY = "diedcity"
    WORK_CITY = "city"
    FIRST_NAME = "firstname"
    SURNAME = "surname"
    MOVIE_TITLE = "Series_Title"
    RELEASE_YEAR_MOVIE = "Released_Year"
    CERTIFICATE_MOVIE = "Certificate"
    GENRE_MOVIE = "Genre"
    IMDB_RATING_MOVIE = "IMDB_Rating"
    VOTES_COUNT_MOVIE = "No_of_Votes"
    DIRECTOR_MOVIE = "Director"
    STAR1_MOVIE = "Star1"
    STAR2_MOVIE = "Star2"
    STAR3_MOVIE = "Star3"
    STAR4_MOVIE = "Star4"
    COUNTRY_NAME = "country_name"
    LEADER_NAME = "leader"
    LEADER_POSITION = "leader_position"




metadata = {
    WIKIBIO: {
        ENTITY: EntityClass.PERSON.value,
        CONTEXTUALISING_ATTRIBUTES: [
            Attribute.NAME.value,
            Attribute.NATIONALITY.value,
            Attribute.OCCUPATION.value
        ],
        TARGET_ATTRIBUTES: {
            ConceptClass.YEAR.value: [Attribute.BIRTH_DATE.value, Attribute.DEATH_DATE.value],
            ConceptClass.PLACE.value: [Attribute.BIRTH_PLACE.value, Attribute.DEATH_PLACE.value]
        }
    },
    NOBEL_PRIZE: {
        ENTITY: EntityClass.NOBEL_LAUREATES.value,
        CONTEXTUALISING_ATTRIBUTES: [
            Attribute.FIRST_NAME.value,
            Attribute.SURNAME.value,
            Attribute.MOTIVATION_NOBEL.value,
            Attribute.CATEGORY_NOBEL.value
        ],
        TARGET_ATTRIBUTES: {
            ConceptClass.YEAR.value: [
                Attribute.BIRTH_DATE_NOBEL.value,
                Attribute.DEATH_DATE_NOBEL.value,
                Attribute.YEAR.value
            ],
            ConceptClass.PLACE.value: [
              Attribute.BIRTH_CITY.value,
              Attribute.DEATH_CITY.value,
              Attribute.WORK_CITY.value
            ]
        }
    },
    MOVIE: {
        ENTITY: EntityClass.MOVIE,
        CONTEXTUALISING_ATTRIBUTES: [
            Attribute.MOVIE_TITLE.value,
            Attribute.RELEASE_YEAR_MOVIE.value,
            Attribute.GENRE_MOVIE.value,
            Attribute.IMDB_RATING_MOVIE.value,
            Attribute.VOTES_COUNT_MOVIE.value,
            Attribute.CERTIFICATE_MOVIE.value
        ],
        TARGET_ATTRIBUTES: {
            ConceptClass.PERSON_NAME.value: [Attribute.DIRECTOR_MOVIE.value,
                                       Attribute.STAR1_MOVIE.value,
                                       Attribute.STAR2_MOVIE.value,
                                       Attribute.STAR3_MOVIE.value,
                                       Attribute.STAR4_MOVIE.value]
        }
    }
}
