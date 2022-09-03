import flwr as fl
from pydantic import BaseModel


class FlServerConfig(BaseModel):
    server_address: str = "0.0.0.0:8080"
    config: fl.server.ServerConfig = dict(num_rounds=3)

    def run(self):
        return fl.server.start_server(
            server_address=self.server_address, config=self.config
        )


# Start Flower server
# fl.server.start_server(
#     server_address="0.0.0.0:8080",
#     config=fl.server.ServerConfig(num_rounds=3),
# )

FlServerConfig().run()
