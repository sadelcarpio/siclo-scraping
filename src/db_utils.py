import os
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT", 5432),
    )


def get_or_create_gym_id(conn, gym_name: str) -> int:
    """Get gym_id by gym_name, inserting if it doesn’t exist."""
    with conn.cursor() as cur:
        cur.execute("""
                    INSERT INTO gimnasios (gym_name)
                    VALUES (%s) ON CONFLICT (gym_name) DO NOTHING
            RETURNING id;
                    """, (gym_name,))

        row = cur.fetchone()
        if row:
            return row[0]

        # If already existed, fetch id
        cur.execute("SELECT id FROM gimnasios WHERE gym_name = %s;", (gym_name,))
        return cur.fetchone()[0]


def bulk_insert(conn, gym_name: str, merged_data: dict):
    """
    Inserts all categories (ubicaciones, precios, horarios, disciplinas) for a gym.
    """
    gym_id = get_or_create_gym_id(conn, gym_name)

    with conn.cursor() as cur:
        # Ubicaciones
        ubicaciones_data = [
            (gym_id, u.get("content_para_busqueda"), u.get("direccion_completa"), u.get("distrito"))
            for u in merged_data.get("ubicaciones", [])
        ]
        execute_batch(cur, """
            INSERT INTO ubicaciones (gym_id, content_para_busqueda, direccion_completa, distrito)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
        """, ubicaciones_data)

        # Precios
        precios_data = [
            (gym_id, p.get("content_para_busqueda"), p.get("sede"), p.get("descripcion_plan"),
             p.get("valor"), p.get("moneda"), p.get("recurrencia"))
            for p in merged_data.get("precios", [])
        ]
        execute_batch(cur, """
            INSERT INTO precios (gym_id, content_para_busqueda, sede, descripcion_plan, valor, moneda, recurrencia)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
        """, precios_data)

        # Horarios
        horarios_data = [
            (gym_id, h.get("sede"), h.get("nombre_clase"), h.get("fecha"), h.get("dia_semana"), h.get("hora_inicio"), h.get("hora_fin"))
            for h in merged_data.get("horarios", [])
        ]
        execute_batch(cur, """
            INSERT INTO horarios (gym_id, sede, nombre_clase, fecha, dia_semana, hora_inicio, hora_fin)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
        """, horarios_data)

        # Disciplinas
        disciplinas_data = [
            (gym_id, d.get("nombre"), d.get("descripcion"))
            for d in merged_data.get("disciplinas", [])
        ]
        execute_batch(cur, """
            INSERT INTO disciplinas (gym_id, nombre, descripcion)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING;
        """, disciplinas_data)

    conn.commit()
    print(f"✅ Bulk inserted all data for gym: {gym_name} (id={gym_id})")
