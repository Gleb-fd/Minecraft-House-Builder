package org.erida.eridasentinel;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Erida_Sentinel extends JavaPlugin {

    @Override
    public void onEnable() {
        getLogger().info("HouseBuilder plugin has been enabled!");
        this.getCommand("buildhouse").setExecutor(new BuildHouseCommand());
    }

    @Override
    public void onDisable() {
        getLogger().info("HouseBuilder plugin has been disabled!");
    }

    class BuildHouseCommand implements CommandExecutor {
        @Override
        public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
            if (sender instanceof Player) {
                Player player = (Player) sender;
                if (args.length < 1) {
                    player.sendMessage("Использовать: /buildhouse <csv_file_name>");
                    return true;
                }
                String fileName = args[0]; // здесь можно доделать так чтобы путь до csv передавался через команду
                buildHouseFromCSV(player.getLocation(), "Путь/до/csv/файла/house_blocks.csv"); // путь до csv файла который сгенерировала t5
                return true;
            }
            return false;
        }

        private void buildHouseFromCSV(Location startLocation, String fileName) {
            try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
                String line;
                boolean firstLine = true;
                while ((line = br.readLine()) != null) {
                    if (firstLine) {
                        firstLine = false;
                        continue;  // Пропускае заголовки
                    }
                    String[] values = line.split(",");
                    if (values.length != 4) {
                        getLogger().warning("Invalid line in CSV: " + line);
                        continue;
                    }
                    int x = Integer.parseInt(values[0]);
                    int y = Integer.parseInt(values[1]);
                    int z = Integer.parseInt(values[2]);
                    Material material = Material.valueOf(values[3].toUpperCase());

                    Location blockLocation = startLocation.clone().add(x, y, z);
                    blockLocation.getBlock().setType(material);
                }
                getLogger().info("House built successfully!");
            } catch (IOException e) {
                getLogger().severe("Error reading CSV file: " + e.getMessage());
            } catch (IllegalArgumentException e) {
                getLogger().severe("Invalid material in CSV file: " + e.getMessage());
            }
        }
    }
}
