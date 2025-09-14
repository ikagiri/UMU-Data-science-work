# Pokémon Analysis & Battle Simulation

# This program uses the Pokémon dataset (from Kaggle).
# It demonstrates:
# Classes and Inheritance (Pokemon, FirePokemon, WaterPokemon, ElectricPokemon)
# Functions (load_data, group_by_type, simulate_battle)
# Loops, Lists, Dictionaries, If statements
# Object-Oriented Programming in Python

# Requirements as per the assignment:
# At least 2 functions
# At least 1 class with a method
# Use of loops, lists, and if statements
# Use of dictionaries to group data

import pandas as pd  # Importing pandas for command prompts

class Pokemon:
	"""
	General class for all Pokémon.
	Stores attributes such as name, type, HP, Attack, Defense, Speed, and Legendary status.
	"""
	def __init__(self, name, p_type, hp, attack, defense, speed, legendary):
		self.name = name
		self.p_type = p_type
		self.hp = hp
		self.attack = attack
		self.defense = defense
		self.speed = speed
		self.legendary = legendary

	def describe(self):
		"""Return a string summary of the Pokémon's stats."""
		return f"{self.name} ({self.p_type}) - HP: {self.hp}, Atk: {self.attack}, Def: {self.defense}, Speed: {self.speed}, Legendary: {self.legendary}"

	def take_damage(self, damage):
		"""
		Reduces HP when attacked.
		Defense reduces the amount of damage taken.
		Returns True if still alive, False if HP reaches zero or below.
		"""
		reduced = max(0, damage - self.defense)  # damage reduced by defense
		self.hp -= reduced
		return self.hp > 0

# Inherited Classes
# These classes add unique special attack methods to demonstrate inheritance and method overriding
class FirePokemon(Pokemon):
	def special_attack(self, opponent):
		damage = int(self.attack * 1.5)  # Fire Pokémon do 1.5x attack
		print(
			f"{self.name} unleashes a Fire Blast on {opponent.name} for {damage} damage!")
		return opponent.take_damage(damage)

class WaterPokemon(Pokemon):
	def special_attack(self, opponent):
		damage = int(self.attack * 1.3)  # Water Pokémon do 1.3x attack
		print(
			f"{self.name} uses a Water Cannon on {opponent.name} for {damage} damage!")
		return opponent.take_damage(damage)

class ElectricPokemon(Pokemon):
	def special_attack(self, opponent):
		damage = int(self.attack * 1.4)  # Electric Pokémon do 1.4x attack
		print(
			f"{self.name} shocks {opponent.name} with Thunderbolt for {damage} damage!")
		return opponent.take_damage(damage)

# Functions
def load_data(file_path):
	"""Load Pokémon data from CSV file and create Pokémon objects."""
	df = pd.read_csv("C:/Users/Admin/Desktop/WORK 2025/MASTERS INFORMATION SYSTEMS WORK/DATA SCIENCE AND VISUALISATION/Pokemon.csv")
	pokemons = []

	# Iterate through dataset rows
	for _, row in df.iterrows():
		p_type = row['Type 1']
		# Use inheritance for certain types
		if p_type == "Fire":
			p = FirePokemon(row['Name'], p_type, row['HP'], row['Attack'],
							row['Defense'], row['Speed'], row['Legendary'])
		elif p_type == "Water":
			p = WaterPokemon(row['Name'], p_type, row['HP'], row['Attack'],
							 row['Defense'], row['Speed'], row['Legendary'])
		elif p_type == "Electric":
			p = ElectricPokemon(row['Name'], p_type, row['HP'], row['Attack'],
								row['Defense'], row['Speed'], row['Legendary'])
		else:
			p = Pokemon(row['Name'], p_type, row['HP'], row['Attack'],
						row['Defense'], row['Speed'], row['Legendary'])
		pokemons.append(p)
	return pokemons

def group_by_type(pokemons):
	"""
	Groups Pokémon by their primary type.
	Returns a dictionary where keys are types and values are lists of Pokémon names.
	"""
	type_groups = {}
	for p in pokemons:
		if p.p_type not in type_groups:
			type_groups[p.p_type] = []
		type_groups[p.p_type].append(p.name)
	return type_groups

def simulate_battle(pokemon1, pokemon2):
	"""
	Simulates a battle between two Pokémon.
	- Faster Pokémon attacks first.
	- Fire, Water, Electric Pokémon use their special attack.
	- Other Pokémon use normal attack.
	Returns the name of the winning Pokémon.
	"""
	print(
		f"\n Battle Start: {pokemon1.name} ({pokemon1.p_type}) vs {pokemon2.name} ({pokemon2.p_type})")

	# Faster Pokémon attacks first
	if pokemon1.speed < pokemon2.speed:
		pokemon1, pokemon2 = pokemon2, pokemon1

	# Keep battling until one faints
	while pokemon1.hp > 0 and pokemon2.hp > 0:
		# Pokemon1 attacks
		if isinstance(pokemon1, (FirePokemon, WaterPokemon, ElectricPokemon)):
			alive = pokemon1.special_attack(pokemon2)
		else:
			alive = pokemon2.take_damage(pokemon1.attack)
		if not alive:
			print(f"{pokemon2.name} fainted! {pokemon1.name} wins!")
			return pokemon1.name

		# Pokemon2 attacks
		if isinstance(pokemon2, (FirePokemon, WaterPokemon, ElectricPokemon)):
			alive = pokemon2.special_attack(pokemon1)
		else:
			alive = pokemon1.take_damage(pokemon2.attack)
		if not alive:
			print(f"{pokemon1.name} fainted! {pokemon2.name} wins!")
			return pokemon2.name

# Main Program
if __name__ == "__main__":
	# 1. Loading Pokémon dataset
	pokemons = load_data(
		"C:/Users/Admin/Desktop/WORK 2025/MASTERS INFORMATION SYSTEMS WORK/DATA SCIENCE AND VISUALISATION/Pokemon.csv")

	# 2. Grouping Pokémon by type (dictionary usage)
	type_groups = group_by_type(pokemons)
	print("Fire Pokémon examples:",
		  type_groups.get("Fire", [])[:5])  # Show first 5 Fire-types

	# 3. Simulating a battle between two Pokémon
	winner = simulate_battle(pokemons[3], pokemons[
		6])  # Example: 4th vs 7th Pokémon in dataset
	print("Winner:", winner)