class MaterialLoader:
    def __init__(self):
        self.materials = {}

    def load_mtl(self, filepath: str):
        current_material = None

        with open(filepath, 'r') as file:
            for line in file:
                if line.startswith('newmtl '):
                    current_material = line.split()[1]
                    self.materials[current_material] = {}
                elif line.startswith('Kd ') and current_material:
                    parts = line.split()
                    self.materials[current_material]['Kd'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                elif line.startswith('Ka ') and current_material:
                    parts = line.split()
                    self.materials[current_material]['Ka'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                elif line.startswith('Ks ') and current_material:
                    parts = line.split()
                    self.materials[current_material]['Ks'] = [float(parts[1]), float(parts[2]), float(parts[3])]

        return self.materials
