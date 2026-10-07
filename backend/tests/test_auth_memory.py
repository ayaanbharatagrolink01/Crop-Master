import os

os.environ.setdefault(
    "JWT_SECRET",
    "test-jwt-secret",
)
os.environ.setdefault(
    "MOBILE_PEPPER",
    "test-pepper",
)
os.environ.setdefault(
    "OTP_PROVIDER",
    "console",
)

import pytest

from app import farmers


@pytest.fixture(autouse=True)
def db_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        farmers,
        "DB_PATH",
        tmp_path / "farmers.db",
    )

    farmers.init_db()

    yield


def test_mobile_normalisation_and_hash():
    from app.auth import (
        normalize_mobile,
        mobile_hash,
    )

    assert (
        normalize_mobile(
            "+91 09876543210"
        )
        == "9876543210"
    )

    assert len(
        mobile_hash("9876543210")
    ) == 64


def test_conversation_isolation():
    from app.farmers import (
        create_conversation,
        create_user,
        get_messages,
        save_message,
    )

    a = create_user(
        "a" * 64,
        "1111",
    )

    b = create_user(
        "b" * 64,
        "2222",
    )

    ca = create_conversation(
        a["id"]
    )

    save_message(
        a["id"],
        ca,
        "user",
        "private",
    )

    assert (
        get_messages(
            b["id"],
            ca,
        )
        is None
    )


def test_case_notes_roundtrip():
    from app.farmers import (
        create_conversation,
        create_user,
        latest_case_notes,
        save_case_notes,
    )

    u = create_user(
        "c" * 64,
        "3333",
    )

    c = create_conversation(
        u["id"]
    )

    save_case_notes(
        u["id"],
        c,
        [
            {
                "crop": "soybean",
                "problem": "yellow mosaic",
                "diagnosis": "virus",
                "advice": "Remove infected plants",
                "refs": ["DIS1"],
            }
        ],
    )

    assert (
        latest_case_notes(
            u["id"]
        )[0]["problem"]
        == "yellow mosaic"
    )


def test_otp_wrong_attempts_lock_and_expiry(
    monkeypatch,
):
    from app.auth import (
        mobile_hash,
        request_otp,
        verify_otp,
    )

    import hashlib
    import sqlite3

    request_otp("9876543210")

    mh = mobile_hash(
        "9876543210"
    )

    con = sqlite3.connect(
        farmers.DB_PATH
    )

    con.execute(
        """
        UPDATE otp_challenges
        SET otp_hash=?,
            expires_at=?
        WHERE mobile_hash=?
        """,
        (
            hashlib.sha256(
                (
                    "123456"
                    + "test-pepper"
                ).encode()
            ).hexdigest(),
            9999999999,
            mh,
        ),
    )

    con.commit()
    con.close()

    for _ in range(4):
        with pytest.raises(Exception):
            verify_otp(
                "9876543210",
                "000000",
            )

    with pytest.raises(Exception):
        verify_otp(
            "9876543210",
            "000000",
        )

    con = sqlite3.connect(
        farmers.DB_PATH
    )

    con.execute(
        """
        UPDATE otp_challenges
        SET expires_at=?,
            wrong_attempts=0,
            locked_until=0
        WHERE mobile_hash=?
        """,
        (
            1,
            mh,
        ),
    )

    con.commit()
    con.close()

    with pytest.raises(Exception):
        verify_otp(
            "9876543210",
            "123456",
        )


def test_otp_request_limit():
    from app.auth import request_otp

    for _ in range(5):
        request_otp(
            "9876543210"
        )

    with pytest.raises(Exception):
        request_otp(
            "9876543210"
        )


def test_cart_data_parser_is_defensive():
    from app.import_orders import (
        parse_cart_products,
    )

    assert (
        parse_cart_products(
            '[{"attr_sku":"Soybean Seed"}]'
        )[0]
        == ["Soybean Seed"]
    )

    assert (
        parse_cart_products(
            '"[{\\"attr_sku\\":\\"Wheat Seed\\"}]"'
        )[0]
        == ["Wheat Seed"]
    )

    assert (
        parse_cart_products(
            '[{prod_id":"P1" | '
            'attr_sku:"Okra-Seeds-by-Agritech" | '
            'qty:"1"}]'
        )[0]
        == ["Okra-Seeds-by-Agritech"]
    )
