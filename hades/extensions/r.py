"""test."""
from vector_field.vector_field import VectorField

f = VectorField(1, 2, 3, [(4.5, 5.5), (6.5, 7.5)])
print(f"class = {f}")
print(f"width = {f.width}")
print(f"height = {f.height}")
print(f.recalculate_map((5.5, 6.6), 7))
