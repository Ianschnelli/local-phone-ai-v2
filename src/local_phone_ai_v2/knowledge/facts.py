from __future__ import annotations

from pathlib import Path

import yaml

from local_phone_ai_v2.knowledge.fact_renderer import (
    render_address,
    render_email,
    render_opening_hours,
    render_phone_number,
)


WEEKDAY_KEYWORDS = {
    "montag": "monday",
    "montags": "monday",
    "dienstag": "tuesday",
    "dienstags": "tuesday",
    "mittwoch": "wednesday",
    "mittwochs": "wednesday",
    "donnerstag": "thursday",
    "donnerstags": "thursday",
    "freitag": "friday",
    "freitags": "friday",
    "samstag": "saturday",
    "samstags": "saturday",
    "sonntag": "sunday",
    "sonntags": "sunday",
}


OPENING_HOURS_KEYWORDS = (
    "offen",
    "geöffnet",
    "öffnungszeit",
    "öffnungszeiten",
    "wann öffnet",
    "wann öffnen",
    "wann schließt",
    "wann schließen",
)


ADDRESS_KEYWORDS = (
    "adresse",
    "anschrift",
    "wo seid ihr",
    "wo sind sie",
    "wo sitzt ihr",
    "wo sitzen sie",
    "standort",
)


PHONE_KEYWORDS = (
    "telefonnummer",
    "telefon",
    "nummer",
    "anrufen",
)


EMAIL_KEYWORDS = (
    "e-mail",
    "email",
    "mailadresse",
    "mail adresse",
)


class StructuredFactProvider:
    def __init__(
        self,
        facts_path: Path,
    ) -> None:
        self._facts = self._load_facts(
            facts_path
        )

    @staticmethod
    def _load_facts(
        facts_path: Path,
    ) -> dict:
        with facts_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "Structured facts must contain "
                "a YAML object."
            )

        return data

    def lookup(
        self,
        query: str,
    ) -> list[str]:
        normalized_query = query.lower().strip()

        facts: list[str] = []

        if self._is_address_query(
            normalized_query
        ):
            address_fact = self._lookup_address()

            if address_fact:
                facts.append(address_fact)

        if self._is_phone_query(
            normalized_query
        ):
            phone_fact = self._lookup_phone()

            if phone_fact:
                facts.append(phone_fact)

        if self._is_email_query(
            normalized_query
        ):
            email_fact = self._lookup_email()

            if email_fact:
                facts.append(email_fact)

        if (
            self._is_opening_hours_query(
                normalized_query
            )
            or self._has_department_and_weekday(
                normalized_query
            )
        ):
            facts.extend(
                self._lookup_opening_hours(
                    normalized_query
                )
            )

        return facts

    @staticmethod
    def _contains_any(
        query: str,
        keywords: tuple[str, ...],
    ) -> bool:
        return any(
            keyword in query
            for keyword in keywords
        )

    def _is_address_query(
        self,
        query: str,
    ) -> bool:
        return self._contains_any(
            query,
            ADDRESS_KEYWORDS,
        )

    def _is_phone_query(
        self,
        query: str,
    ) -> bool:
        return self._contains_any(
            query,
            PHONE_KEYWORDS,
        )

    def _is_email_query(
        self,
        query: str,
    ) -> bool:
        return self._contains_any(
            query,
            EMAIL_KEYWORDS,
        )

    def _is_opening_hours_query(
        self,
        query: str,
    ) -> bool:
        return self._contains_any(
            query,
            OPENING_HOURS_KEYWORDS,
        )

    def _lookup_address(
        self,
    ) -> str | None:
        contact = self._facts.get(
            "contact",
            {}
        )

        address = contact.get(
            "address"
        )

        if not isinstance(address, dict):
            return None

        return render_address(
            address
        )

    def _lookup_phone(
        self,
    ) -> str | None:
        contact = self._facts.get(
            "contact",
            {}
        )

        phone = contact.get(
            "phone"
        )

        if not isinstance(phone, str):
            return None

        if not phone.strip():
            return None

        return render_phone_number(
            phone
        )

    def _lookup_email(
        self,
    ) -> str | None:
        contact = self._facts.get(
            "contact",
            {}
        )

        email = contact.get(
            "email"
        )

        if not isinstance(email, str):
            return None

        if not email.strip():
            return None

        return render_email(
            email
        )


    def _has_department_and_weekday(
        self,
        query: str,
    ) -> bool:
        if self._resolve_weekday(query) is None:
            return False

        departments = self._facts.get(
            "departments",
            {},
        )

        if not isinstance(departments, dict):
            return False

        matches = self._resolve_departments(
            query,
            departments,
            fallback_to_all=False,
        )

        return bool(matches)

    def _lookup_opening_hours(
        self,
        query: str,
    ) -> list[str]:
        weekday = self._resolve_weekday(
            query
        )

        departments = self._facts.get(
            "departments",
            {}
        )

        if not isinstance(
            departments,
            dict,
        ):
            return []

        requested_departments = (
            self._resolve_departments(
                query,
                departments,
            )
        )

        facts: list[str] = []

        for department_key in (
            requested_departments
        ):
            department = departments.get(
                department_key
            )

            if not isinstance(
                department,
                dict,
            ):
                continue

            label = department.get(
                "label",
                department_key,
            )

            opening_hours = department.get(
                "opening_hours",
                {},
            )

            if not isinstance(
                opening_hours,
                dict,
            ):
                continue

            if weekday is not None:
                if weekday not in opening_hours:
                    continue

                facts.append(
                    render_opening_hours(
                        department_label=label,
                        weekday=weekday,
                        hours=opening_hours[weekday],
                    )
                )

                continue

            for day, hours in opening_hours.items():
                facts.append(
                    render_opening_hours(
                        department_label=label,
                        weekday=day,
                        hours=hours,
                    )
                )

        return facts

    @staticmethod
    def _resolve_weekday(
        query: str,
    ) -> str | None:
        for keyword, weekday in (
            WEEKDAY_KEYWORDS.items()
        ):
            if keyword in query:
                return weekday

        return None

    @staticmethod
    def _resolve_departments(
        query: str,
        departments: dict,
        fallback_to_all: bool = True,
    ) -> list[str]:
        matches: list[str] = []

        for key, department in departments.items():
            if not isinstance(department, dict):
                continue

            label = str(
                department.get(
                    "label",
                    key,
                )
            ).lower()

            aliases = {
                key.lower(),
                label,
            }

            configured_aliases = department.get(
                "aliases",
                [],
            )

            if isinstance(configured_aliases, list):
                aliases.update(
                    str(alias).lower()
                    for alias in configured_aliases
                    if str(alias).strip()
                )

            if any(
                alias in query
                for alias in aliases
            ):
                matches.append(key)

        if matches:
            return matches

        if fallback_to_all:
            return list(departments.keys())

        return []