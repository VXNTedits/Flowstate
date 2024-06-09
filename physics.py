import glm


class Physics:
    def __init__(self, world, player):
        self.world = world
        self.player = player
        self.colliders = []
        self.add_model_collider(world)
        self.add_model_collider(player.model)

    def add_collider(self, collider):
        self.colliders.append(collider)

    def add_model_collider(self, model):
        bounding_box = (model.min_point, model.max_point)
        self.add_collider(bounding_box)

    def check_collision(self, player):
        player_box = self.get_bounding_box(player.position)
        for collider in self.colliders:
            if collider == (player.model.min_point, player.model.max_point):
                continue  # Skip collision check for the player itself
            if self.aabb_collision(player_box, collider):
                self.resolve_collision(player, collider)

    def get_bounding_box(self, position, size=(1.0, 1.0, 1.0)):
        half_size = glm.vec3(size[0] / 2, size[1] / 2, size[2] / 2)
        min_point = position - half_size
        max_point = position + half_size
        return (min_point, max_point)

    def aabb_collision(self, box1, box2):
        (min1, max1) = box1
        (min2, max2) = box2
        return (min1.x <= max2.x and max1.x >= min2.x) and \
               (min1.y <= max2.y and max1.y >= min2.y) and \
               (min1.z <= max2.z and max1.z >= min2.z)

    def resolve_collision(self, player, collider):
        player.position -= player.velocity

    def get_colliders(self):
        return self.colliders

