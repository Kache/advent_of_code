#!/usr/bin/env ruby

require 'bundler/inline'

gemfile do
  source 'https://rubygems.org'
  ruby '~> 2.6.3'
  gem 'pry-byebug'
  gem 'activesupport'
end

require 'active_support/core_ext/enumerable'

module Day1
  @@input = File.readlines('input1').map(&:to_i).freeze

  def self.fuel_req(masses = @@input)
    masses.map { |i| i / 3 - 2 }.reduce(:+)
  end

  def self.fuel_for_fuel(masses = @@input)
    total_fuel = 0

    until masses.empty?
      masses = masses.map { |i| i / 3 - 2 }.select(&:positive?)
      total_fuel += masses.reduce(0, :+)
    end

    total_fuel
  end
end

module Day2
  @@input = File.read('input2').split(',').map(&:to_i).freeze

  def self.exec(intcode = @@input, input = 1202)
    intcode = intcode.dup
    intcode[1] = input / 100 # noun
    intcode[2] = input % 100 # verb

    ptr = 0
    ops = { 1 => :+, 2 => :* }

    loop do
      opcode, x_ref, y_ref, out = intcode[ptr, 4]
      break if opcode == 99

      op = ops.fetch(opcode)
      intcode[out] = intcode[x_ref].send(op, intcode[y_ref])
      ptr += 4
    end

    intcode.first
  end

  def self.find_input(intcode = @@input, output = 19690720)
    (0..9999).each do |input|
      out = compute(intcode, input) rescue -1
      return input if out == output
    end
  end
end

module Day3
  require 'matrix'

  @@input = File.readlines('input3').map do |path|
    path.split(',').map do |dir_dist|
      [dir_dist[0].to_sym, dir_dist[1..-1].to_i].freeze
    end.freeze
  end.freeze

  @@vectors = { U: Vector[1, 0], D: Vector[-1, 0], L: Vector[0, -1], R: Vector[0, 1] }

  def self.wirecross_dist(wirepaths = @@input)
    first_pos_steps = walk_path(wirepaths.first).to_h

    walk_path(wirepaths.last).reduce(Float::INFINITY) do |closest_cross, (pos, steps)|
      cross_dist = first_pos_steps.include?(pos) ? pos[0].abs + pos[1].abs : Float::INFINITY
      [closest_cross, cross_dist].min
    end
  end

  def self.wirecross_steps(wirepaths = @@input)
    first_pos_steps = walk_path(wirepaths.first).to_h

    walk_path(wirepaths.last).reduce(Float::INFINITY) do |closest_cross, (pos, steps)|
      cross_dist = first_pos_steps.fetch(pos, Float::INFINITY) + steps
      [closest_cross, cross_dist].min
    end
  end

  def self.walk_path(wirepath)
    return enum_for(:walk_path, wirepath) unless block_given?

    pos = Vector[0, 0]
    position_steps = Hash.new
    unit_steps = wirepath.flat_map { |dir, dist| Array.new(dist, dir) }

    unit_steps.each.with_index(1) do |step_dir, step_num|
      pos += @@vectors[step_dir]
      position_steps[pos] ||= step_num
      yield pos, position_steps[pos]
    end
  end
end

module Day4
  @@input = Range.new(*'108457-562041'.split('-').map(&:to_i))

  def self.valid_nums(range = @@input)
    range.count do |n|
      valid = (0..4).any? { |e| (((n / 10**e) % 100) % 11).zero? } # two adjacent digits are the same
      valid &&= n.digits.reverse == n.digits.sort
    end
  end

  def self.valid_nums_strict_pairs(range = @@input)
    range.count do |n|
      valid = n.digits.reverse == n.digits.sort
      valid &&= n.digits.chunk(&:itself).any? { |_v, vals| vals.size == 2 } # has a pair that isn't part of a larger group of matching digits
    end
  end
end

class Intcode
  OPS   = { add: 1, mul: 2, inp: 3, out: 4, jit: 5, jif: 6, lt: 7, eq: 8, rbo: 9, hlt: 99 }.invert
  ARITY = { add: 3, mul: 3, inp: 1, out: 1, jit: 2, jif: 2, lt: 3, eq: 3, rbo: 1, hlt: 0  }.transform_keys(&OPS.invert)

  Param = Struct.new(:ref, :val)

  def initialize(intcode)
    @ptr, @relbase = 0, 0
    @intcode = String === intcode ? intcode.split(',').map(&:to_i) : intcode.dup
  end

  def run(input = [])
    while @ptr
      modes, opcode = @intcode.fetch(@ptr).divmod(100)
      raw_params = Array.new(ARITY.fetch(opcode)) { |i| @intcode.fetch(@ptr + 1 + i, 0) }

      a, b, c = raw_params.zip(modes.digits).map do |raw, mode|
        case %i[pos imm rel].fetch(mode || 0)
        when :pos then Param.new(raw,            @intcode.fetch(raw, 0))
        when :imm then Param.new(nil,            raw)
        when :rel then Param.new(@relbase + raw, @intcode.fetch(@relbase + raw, 0))
        end
      end

      @ptr += 1 + ARITY[opcode]
      case OPS[opcode]
      when :add then @intcode[c.ref] = a.val + b.val
      when :mul then @intcode[c.ref] = a.val * b.val
      when :inp then @intcode[a.ref] = input.shift
      when :out then block_given? ? (yield a.val) : (return a.val)
      when :jit then @ptr = b.val unless a.val.zero?
      when :jif then @ptr = b.val if     a.val.zero?
      when :lt  then @intcode[c.ref] = a.val < b.val ? 1 : 0
      when :eq  then @intcode[c.ref] = a.val == b.val ? 1 : 0
      when :rbo then @relbase += a.val
      when :hlt then @ptr = nil
      end
    end
  end
end

module Day5
  @@input = File.read('input5').split(',').map(&:to_i).freeze

  def self.test_air_conditioner
    Intcode.new(@@input).to_enum(:run, [1]).find { |out| !out.zero? }
  end

  def self.test_thermal_radiator
    Intcode.new(@@input).run([5])
  end
end

module Day6
  @@input = File.read('input6').split.map do |orbit|
    orbit.split(')').reverse.each(&:freeze)
  end.to_h.freeze

  def self.num_orbits(orbits = @@input)
    bfs_distances(orbits, 'COM').each_value.reduce(:+)
  end

  def self.num_transfers(orbits = @@input)
    bfs_distances(orbits, 'YOU')['SAN'] - 2
  end

  def self.bfs_distances(edges, start)
    neighbors = Hash.new
    edges.each do |from, to|
      (neighbors[from] ||= []) << to
      (neighbors[to] ||= []) << from
    end

    distances = { start => 0 }
    deque = [start]
    until deque.empty?
      node = deque.shift
      neighbors[node].each do |n|
        deque << n unless distances.include?(n)
        distances[n] ||= distances[node] + 1
      end
    end
    distances
  end
end

module Day7
  @@input = File.read('input7').split(',').map(&:to_i)

  def self.max_signal(intcode = @@input)
    Array(0..4).permutation.map do |phase_settings|
      phase_settings.reduce(0) do |input, setting|
        Intcode.new(intcode).run([setting, input])
      end
    end.max
  end

  def self.max_looped_signal_threaded(intcode = @@input)
    Array(5..9).permutation.map do |phase_settings|
      queues = Array.new(5) { |i| Queue.new << phase_settings[i] }
      queues.first << 0

      Array.new(5) do |i|
        Thread.new do
          Intcode.new(intcode).run(queues[i]) { |v| queues.rotate[i] << v }
        end
      end.each(&:join)
      queues.first.pop
    end.max
  end

  def self.max_looped_signal(intcode = @@input)
    Array(5..9).permutation.map do |phase_settings|
      amps = Array.new(5) { Intcode.new(intcode) }
      buffers = phase_settings.map { |s| [s] }

      amps.zip(buffers).cycle.reduce(0) do |input, (amp, buff)|
        amp.run(buff << input) || (break input)
      end
    end.max
  end
end

module Day8
  @@input = File.read('input8').chomp
  @@w, @@h = 25, 6
  @@printable = { '0' => ' ', '1' => '█', '2' => nil }

  def self.blankness_of_blankest_layer(pixels = @@input)
    pixels.chars.each_slice(@@w * @@h).min_by { |l| l.count('0') }.then { |l| l.count('1') * l.count('2') }
  end

  def self.print_password(pixels = @@input)
    image = Array.new(@@w * @@h)
    pixels.each_char.with_index do |pixel, i|
      image[i % image.size] ||= @@printable[pixel]
    end
    image.each_slice(@@w) { |row| puts row.join }
  end
end

module Day9
  @@input = File.read('input9').split(',').map(&:to_i)

  def self.boost_keycode(intcode = @@input)
    Intcode.new(intcode).to_enum(:run, [1]).to_a
  end

  def self.boost(intcode = @@input)
    Intcode.new(intcode).run([2])
  end
end

module Day10
  @@input = File.read('input10')

  def self.num_viewable(input = @@input)
    _laser_coord, num_bearings = all_obj_bearings(input).transform_values(&:size).max_by(&:last)
    num_bearings
  end

  def self.nth_vaporized(input = @@input, n = 200)
    all_bearings = all_obj_bearings(input)
    laser_coord, _num_bearings = all_bearings.transform_values(&:size).max_by(&:last)

    ordered_targets = all_bearings[laser_coord].flat_map do |bearing, group|
      turn_indexed = group.sort_by { |target| distance(*laser_coord, *target) }.each_with_index
      turn_indexed.map { |target, turn_num| [turn_num * 2 * Math::PI + bearing, target] }
    end.sort

    _radians, (x, y) = ordered_targets[n - 1]
    x * 100 + y
  end

  def self.all_obj_bearings(input)
    asteroids = input.each_line.with_index.flat_map do |line, y|
      line.each_char.with_index.map { |c, x| [x, y] if c == '#' }
    end.compact

    asteroids.index_with do |candidate|
      asteroids.without([candidate]).group_by { |asteroid| laser_angle(*candidate, *asteroid) }
    end
  end

  def self.distance(from_x, from_y, to_x, to_y)
    Math.sqrt((from_x - to_x)**2 + (from_y - to_y)**2)
  end

  def self.laser_angle(from_x, from_y, to_x, to_y)
    y, x = to_x - from_x, -(to_y - from_y) # coordinate conversion
    Math.atan2(y, x) % (2 * Math::PI)
  end
end

module Day11
  COLOR = { black: 0, white: 1 }
  PAINT = { black: ' ', white: '█' }.transform_keys(&COLOR)

  def self.painting_estimate
    paint(COLOR[:black]).size
  end

  def self.paint_letters
    painted = paint(COLOR[:white])

    dim_x, dim_y = painted.keys.transpose.map(&:minmax).map { |lo, hi| (lo..hi) }
    dim_y.map do |y|
      dim_x.map { |x| PAINT[painted.fetch([x, y], COLOR[:black])] }.join
    end.join("\n")
  end

  def self.paint(hull_colorcode, intcode = File.read('input11'))
    robot = Intcode.new(intcode)
    coord = Vector[0, 0]
    facing = [Vector[0, -1], Vector[1, 0], Vector[0, 1], Vector[-1, 0]] # coordinate conversion
    painted = Hash.new

    while (paintcode = robot.run([hull_colorcode]))
      painted[coord.to_a] = paintcode
      coord += facing.rotate!([-1, 1].at(robot.run)).first
      hull_colorcode = painted.fetch(coord.to_a, COLOR[:black])
    end
    painted
  end
end

module Day12
  @@input = File.read('input12').scan(/<x=(.*?), y=(.*?), z=(.*?)>/).map { |nums| nums.map(&:to_i) }

  Moon = Struct.new(:coord, :veloc)

  def self.total_energy(coords = @@input)
    moons = coords.map { |coord| Moon.new(Vector[*coord], Vector[0, 0, 0]) }

    1000.times do
      moons.combination(2).each do |a, b|
        (0..2).each do |axis|
          grav = a.coord[axis] <=> b.coord[axis]
          a.veloc[axis] -= grav
          b.veloc[axis] += grav
        end
      end

      moons.each { |m| m.coord += m.veloc }
    end

    moons.map { |m| m.map { |v| v.map(&:abs).sum }.inject(:*) }.sum
  end

  def self.universe_period(coords = @@input)
    coords.transpose.map do |positions|
      moons = positions.map { |p| [p, 0] }

      (1..).each do |step|
        moons.combination(2).each do |a, b|
          grav = a[0] <=> b[0]
          a[1] -= grav
          b[1] += grav
        end
        moons.each { |m| m[0] += m[1] }

        break step * 2 if moons.all? { |p, v| v.zero? } # universe timeline is "symmetric"
      end
    end.reduce(:lcm)
  end
end

module Day13
  @@input = File.read('input13')

  OBJ = { empty: 0, wall: 1, block: 2, paddle: 3, ball: 4 }
  TILE = { empty: ' ', wall: '█', block: '□', paddle: '▔', ball: '•' }.transform_keys(&OBJ)

  def self.num_blocks(intcode = @@input.split(',').map(&:to_i))
    output, game = [], Intcode.new(intcode); nil
    game.run { |o| output << o }
    output.each_slice(3).count { |x, y, t| t == OBJ[:block] }
    output.each_slice(3).map(&:last).each_slice(44).each { |row| puts row.map(&TILE).join }
  end

  def self.play(intcode = @@input.split(',').map(&:to_i))
    game = Intcode.new([2, *intcode[1..-1]]) # play for free
    board = Array.new(20) { Array.new(44) }
    board[0] << " Score: " << 0
    input, ball_pos, paddle = [], [-1, -1], 99

    while (x, y, t = Array.new(3) { game.run(input) }).all?
      if t == OBJ[:paddle]
        paddle = x
      elsif t == OBJ[:ball]
        ball_pos = [x, y]
        input << (ball_pos[0] <=> paddle)
      elsif [x, y] == ball_pos # frame
        board.each { |row| puts row.join }
        sleep 0.05
      end

      board[y][x] = x.negative? ? t : TILE[t]
    end
    board.each { |row| puts row.join } && board[0][-1]
  end
end

module Day14
  @@input = File.read('input14')

  @@test = <<~TEST
    9 ORE => 2 A
    8 ORE => 3 B
    7 ORE => 5 C
    3 A, 4 B => 1 AB
    5 B, 7 C => 1 BC
    4 C, 1 A => 1 CA
    2 AB, 3 BC, 4 CA => 1 FUEL
  TEST

  @@test = <<~TEST
    10 ORE => 10 A
    1 ORE => 1 B
    7 A, 1 B => 1 C
    7 A, 1 C => 1 D
    7 A, 1 D => 1 E
    7 A, 1 E => 1 FUEL
  TEST

  <<~RUN
    7 A, 1 E
  RUN

  require 'tsort'

  def self.fuel(input = @@test, fuel = 1)
    recipes = parse(input)

    graph = recipes.map { |prod, (num, reag)| [prod, reag.map(&:first)] }.to_h
    graph["ORE"] = []

    graph.singleton_class.class_eval do
      include TSort
      alias_method :tsort_each_node, :each_key

      def tsort_each_child(node, &block)
        fetch(node).each(&block)
      end
    end

    req_order = graph.tsort.each_with_index.to_h



    required = { "FUEL" => fuel }

    until required.keys == ["ORE"]
      req = required.each_key.max_by { |k| req_order[k] }
      req_cnt = required.delete(req)

      makes, reageants = recipes.fetch(req)
      reageants.each do |reag, cnt|
        required[reag] ||= 0
        required[reag] += cnt * (req_cnt.to_f / makes).ceil
      end
    end

    required["ORE"]
  end

  # (0..5000000).to_a.bsearch { |n| fuel(@@input, n) > 1_000_000_000_000 }

  def self.parse(input = @@test)
    input.each_line.map do |line|
      reageants_str, product = line.split('=>')
      (prod_cnt, prod), *reageants = [product, *reageants_str.split(',')].map do |ingred|
        count, name = ingred.split(' ')
        [count.to_i, name]
      end
      [prod, [prod_cnt, reageants.map(&:reverse).to_h]]
    end.to_h
  end
end

module Day15
  @@input = File.read('input15')

  DIR = { north: 1, south: 2, west: 3, east: 4 }
  VEC = { north: Vector[0, -1], south: Vector[0, 1], west: Vector[-1, 0], east: Vector[1, 0] }
  STATUS = { hit_wall: 0, moved: 1, found_oxy: 2 }.invert
  OBJ = { empty: ' ', wall: '█', oxy: '□' }

  def self.explore(input = @@input)

    droid = Intcode.new(input); nil
    world = { [0, 0] => 'O' }
    map = Hash.new { |h, k| h[k] = Set.new }
    pos = Vector[0, 0]
    oxy_pos = nil
    heading = %i[north east south west]
    go = lambda do |dir|
      status = STATUS.fetch(droid.run([DIR.fetch(dir)]))
      case status
      when :hit_wall
        world[(pos + VEC[dir]).to_a] = OBJ[:wall]
      when :moved
        map[pos] << pos + VEC[dir]
        map[pos + VEC[dir]] << pos
        pos += VEC[dir]
        world[pos.to_a] = OBJ[:empty]
      when :found_oxy
        map[pos] << pos + VEC[dir]
        map[pos + VEC[dir]] << pos
        pos += VEC[dir]
        finish = pos.to_a
        world[pos.to_a] = OBJ[:oxy]
      when nil
        :done
      end
      status
    end

    # path = lambda do |dlist|
    #   dlist.each do |d|
    #     10.times { ->(dir) { go[dir]; print(world, pos.to_a) }.call SHORT[d] }
    #   end
    # end

    directed = lambda do
      loop do
        break if oxy_pos && pos.zero?
        status = go[heading.first]
        case status
        when :hit_wall
          heading.rotate!
        when :moved
          heading.rotate!(-1)
        when :found_oxy
          oxy_pos = pos
          # puts "Found Oxy"
        #   break
        # when :done
        #   break
        end
        print(world, pos.to_a)
      end
    end

    directed.call

    distances = {}
    deque = [[Vector[*oxy_pos], 0]]
    until deque.empty?
      curr, dist = deque.shift
      distances[curr] = dist

      neighbors = map[Vector[*curr]]
      neighbors.reject(&distances).each do |n|
        deque << [n, dist + 1]
      end
    end

    a = distances[Vector[0, 0]] # part 1
    b = distances.each_value.max # part 2
    [a, b]
  end

  def self.print(world, pos)
    puts
    dim_x, dim_y = world.keys.transpose.map(&:minmax).map { |lo, hi| (lo..hi) }
    dim_y.each do |y|
      puts dim_x.map { |x| pos.to_a == [x, y] ? 'X' : world.fetch([x, y], OBJ[:empty]) }.join
    end
    sleep 0.05
  end
end

module Day16
  @@input = File.read('input16')

  def self.fft(input = @@input.strip)
    nums, base_pat = input.chars.map(&:to_i), [0, 1, 0, -1]

    100.times do
      (1..nums.size).map do |nth|
        nums[nth - 1] = (nth..nums.size).reduce(0) do |sum, n|
          sum + nums[n - 1] * base_pat[(n / nth) % 4]
        end.abs % 10
      end
    end
    nums.first(8).join
  end

  def self.fft_offset(input = @@input.strip)
    partial, offset = input.chars.map(&:to_i), input[0, 7].to_i
    full = (partial * 10_000)[offset..-1]

    100.times do
      (full.size - 2).downto(0) do |i|
        full[i] = (full[i] + full[i + 1]) % 10
      end
    end
    full.first(8).join
  end
end
